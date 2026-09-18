import numpy as np
import pandas as pd
from typing import Callable, Literal
from workshop_infrastructure.datasets.helio import HelioNetCDFDataset


class FilamentDataset(HelioNetCDFDataset):
    """
    Template child class of HelioNetCDFDataset showing how to build a downstream dataset.
    Extends the base class with a filament chirality label (0 = dextral, 1 = sinistral)
    aligned to the Surya index. ``hemisphere`` is also carried through as passthrough
    metadata — it drives the Martin's Rule baseline, but is never fed to the model as an
    input feature, since that would hand it the answer this comparison is meant to test for.

    All ``HelioNetCDFDataset`` keyword arguments (``index_path``, ``scalers``, ``channels``,
    ``s3_cache_dir``, etc.) are accepted via ``**kwargs`` and forwarded to the base class.
    ``load_forecast_frames`` defaults to ``False`` here (filament chirality supplies its own
    labels, so future Surya frames are never fetched); pass it explicitly to override.

    Additional Args:
        return_surya_stack: If True (default), include the Surya image stack in the returned dict.
            Set to False to return only the flare intensity label (useful for label inspection).
        max_number_of_samples: Cap the dataset length at this value. Useful for quick experiments.
        label_transform: Optional callable applied to the ``intensity`` column of the flare index
            to produce the ``normalized_intensity`` label. Signature:
            ``(series: pd.Series) -> pd.Series``.  If ``None``, the raw intensity values are
            used as-is. Define this at the call site (e.g., in ``build_datasets()``) to keep
            normalization logic out of the dataset class.
        filament_index_path: Path to the downstream filament chirality CSV index. If the CSV
            has a ``split`` column, only rows whose value equals this dataset's ``phase``
            ("train" / "val") are used, so the train/val split is defined by the catalog
            rather than by which Surya index file is read. Without that column the whole
            catalog is used, and the split must come from the Surya indices instead.
        ds_time_column: Column name in the flare index to use as the event timestamp.
        ds_time_tolerance: Maximum allowed time offset when matching Surya and DS indices
            (e.g., ``"15min"``). Unmatched entries are dropped.
        ds_match_direction: Merge direction passed to ``pd.merge_asof``. Use ``"forward"``
            for causal prediction (predict flares from prior solar state).

    Raises:
        ValueError: If ``filament_index_path`` is not provided, if the catalog has a
            ``split`` column with no rows matching this dataset's ``phase``, or if no
            overlap exists between the Surya and DS indices within the specified tolerance.
    """

    def __init__(
        self,
        # Downstream-specific parameters
        return_surya_stack: bool = True,
        max_number_of_samples: int | None = None,
        label_transform: Callable[[pd.Series], pd.Series] | None = None,
        filament_index_path: str | None = None,
        ds_time_column: str | None = None,
        ds_time_tolerance: str | None = None,
        ds_match_direction: Literal["forward", "backward", "nearest"] = "forward",
        # All HelioNetCDFDataset parameters (index_path, scalers, channels, s3_*, etc.)
        **kwargs,
    ):
        if ds_match_direction not in ["forward", "backward", "nearest"]:
            raise ValueError("ds_match_direction must be one of 'forward', 'backward', or 'nearest'")

        # load_forecast_frames defaults to False here: flare forecasting supplies its
        # own labels, so future Surya frames never need to be fetched from disk/S3.
        kwargs.setdefault("load_forecast_frames", False)
        super().__init__(**kwargs)

        self.return_surya_stack = return_surya_stack

        # Load ds index and find intersection with Surya index
        if filament_index_path is not None:
            self.ds_index = pd.read_csv(filament_index_path)
        else:
            raise ValueError("filament_index_path must be provided for FilamentDataset")

        # The train/val split comes from the catalog's ``split`` column, not from the Surya
        # index. The shipped Surya indices are carved by month and exclude 2012 entirely,
        # where most of this catalog lives, so both ``train_data_path`` and
        # ``valid_data_path`` point at the full index — which means this filter is the only
        # thing keeping the two datasets from being identical. The column is optional so
        # that a catalog whose events do span the shipped index months still works.
        if "split" in self.ds_index.columns:
            available = sorted(self.ds_index["split"].dropna().unique())
            self.ds_index = self.ds_index.loc[
                self.ds_index["split"] == self.phase, :
            ].copy()
            if len(self.ds_index) == 0:
                raise ValueError(
                    f"Filament catalog {filament_index_path} has no rows with "
                    f"split == '{self.phase}'; split values present: {available}"
                )

        # chirality is the prediction target. hemisphere is kept as passthrough metadata
        # only (see __getitem__) for the Martin's Rule baseline.
        if not self.ds_index["chirality"].isin([0, 1]).all():
            raise ValueError("chirality column must contain only 0 (dextral) or 1 (sinistral)")
        self.ds_index["label"] = self.ds_index["chirality"].astype(np.float32)

        if self.ds_index["label"].isna().any():
            raise ValueError("chirality must be either 0 (dextral) or 1 (sinistral)")

        self.ds_index["ds_index"] = pd.to_datetime(
            self.ds_index[ds_time_column]
        ).values.astype("datetime64[ns]")
        self.ds_index.sort_values("ds_index", inplace=True)

        # Create Surya valid indices and find closest match to DS index
        self.df_valid_indices = pd.DataFrame(
            {"valid_indices": self.valid_indices}
        ).sort_values("valid_indices")
        self.df_valid_indices = pd.merge_asof(
            self.df_valid_indices,
            self.ds_index,
            right_on="ds_index",
            left_on="valid_indices",
            direction=ds_match_direction,
        )
        # Remove duplicates keeping closest match
        self.df_valid_indices["index_delta"] = np.abs(
            self.df_valid_indices["valid_indices"] - self.df_valid_indices["ds_index"]
        )
        self.df_valid_indices = self.df_valid_indices.sort_values(
            ["ds_index", "index_delta"]
        )
        self.df_valid_indices.drop_duplicates(
            subset="ds_index", keep="first", inplace=True
        )
        # Enforce a maximum time tolerance for matches
        if ds_time_tolerance is not None:
            self.df_valid_indices = self.df_valid_indices.loc[
                self.df_valid_indices["index_delta"] <= pd.Timedelta(ds_time_tolerance),
                :,
            ]
            if len(self.df_valid_indices) == 0:
                raise ValueError("No intersection between Surya and DS indices")

        # Override valid indices variables to reflect matches between Surya and DS
        self.valid_indices = [
            pd.Timestamp(date) for date in self.df_valid_indices["valid_indices"]
        ]
        self.adjusted_length = len(self.valid_indices)
        self.df_valid_indices.set_index("valid_indices", inplace=True)

        if max_number_of_samples is not None and max_number_of_samples < self.adjusted_length:
            self.valid_indices = self.valid_indices[:max_number_of_samples]
            self.df_valid_indices = self.df_valid_indices.iloc[:max_number_of_samples]
            self.adjusted_length = max_number_of_samples

    def __len__(self):
        return self.adjusted_length

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: Dataset index.

        Returns:
            Dictionary containing:
                forecast (np.float32): Chirality label (0 = dextral, 1 = sinistral).
                hemisphere (int): Passthrough metadata (0/1), not a model input — used for
                    the Martin's Rule baseline and for reporting accuracy split by hemisphere.
                ds_index (str): ISO-format timestamp from the filament index.
            When ``return_surya_stack=True``, also includes all keys from
            ``HelioNetCDFDataset.__getitem__`` (ts, time_delta_input, lead_time_delta, etc.).
        """
        sample = super().__getitem__(idx=idx) if self.return_surya_stack else {}
        sample["forecast"] = np.float32(self.df_valid_indices.iloc[idx]["label"])
        sample["hemisphere"] = int(self.df_valid_indices.iloc[idx]["hemisphere"])
        sample["ds_index"] = self.df_valid_indices["ds_index"].iloc[idx].isoformat()
        return sample



