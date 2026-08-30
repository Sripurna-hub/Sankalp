# =============================================================================
# backend/csv_registry.py
#
# Central CSV -> image location resolver
#
# Supports:
#   Image_Path / image_path / Image Path
#   Filename / File_Name / file_name
#   Image / Image_Name
#   Latitude / latitude / Lat
#   Longitude / longitude / Lon
#   Event / event
#
# Matching:
#   1. Exact filename
#   2. Basename
#   3. Normalized filename
#   4. Filename stem
#
# This is shared by PUBLIC and INFRASTRUCTURE uploads.
# =============================================================================

import os
import re
import pandas as pd


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def _normalize_text(value):
    """
    Normalize text so CSV paths and uploaded filenames can be compared safely.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().strip('"').strip("'")

    # Convert Windows paths to normal separators.
    text = text.replace("\\", "/")

    # Remove repeated spaces.
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def _normalize_filename(value):
    """
    Normalize only the filename portion.
    """

    text = _normalize_text(value)

    if not text:
        return ""

    return os.path.basename(text)


def _filename_stem(value):
    """
    Return filename without extension.
    """

    filename = _normalize_filename(value)

    if not filename:
        return ""

    return os.path.splitext(filename)[0]


# =============================================================================
# COLUMN FINDER
# =============================================================================

def _find_column(df, aliases):
    """
    Find a CSV column using several possible names.

    Example:
        Image_Path
        image_path
        Image Path
        filename
        File_Name
    """

    normalized_columns = {}

    for column in df.columns:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            str(column).strip().lower()
        )
        normalized_columns[key] = column

    for alias in aliases:

        alias_key = re.sub(
            r"[^a-z0-9]",
            "",
            alias.strip().lower()
        )

        if alias_key in normalized_columns:
            return normalized_columns[alias_key]

    return None


# =============================================================================
# MAIN LOOKUP
# =============================================================================

def lookup_coordinates_from_csv(file_name, df_metadata):
    """
    Find latitude, longitude and event for an uploaded image.

    The same function is used by:
        - Public Tab
        - Infrastructure Tab
        - Bulk Infrastructure Dataset

    Returns:
        (latitude, longitude, event)

    If no match:
        (None, None, None)
    """

    if df_metadata is None:
        return None, None, None

    if not isinstance(df_metadata, pd.DataFrame):
        return None, None, None

    if df_metadata.empty:
        return None, None, None

    # -------------------------------------------------------------------------
    # Clean column names
    # -------------------------------------------------------------------------

    df = df_metadata.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # -------------------------------------------------------------------------
    # Find relevant columns.
    #
    # We support multiple CSV formats.
    # -------------------------------------------------------------------------

    image_column = _find_column(
        df,
        [
            "Image_Path",
            "Image Path",
            "imagepath",
            "Image",
            "Image_Name",
            "Image Name",
            "Filename",
            "File_Name",
            "file_name",
            "File",
            "File Path",
            "Path",
            "image",
        ]
    )

    latitude_column = _find_column(
        df,
        [
            "Latitude",
            "latitude",
            "Lat",
            "lat",
            "GPS_Latitude",
            "GPS Latitude",
        ]
    )

    longitude_column = _find_column(
        df,
        [
            "Longitude",
            "longitude",
            "Lon",
            "lon",
            "Long",
            "GPS_Longitude",
            "GPS Longitude",
        ]
    )

    event_column = _find_column(
        df,
        [
            "Event",
            "event",
            "Disaster",
            "Disaster Event",
            "Event_Name",
            "Event Name",
        ]
    )

    # We cannot locate coordinates without these.
    if image_column is None:
        return None, None, None

    if latitude_column is None or longitude_column is None:
        return None, None, None

    # -------------------------------------------------------------------------
    # Normalize uploaded filename.
    # -------------------------------------------------------------------------

    uploaded_full = _normalize_text(file_name)
    uploaded_name = _normalize_filename(file_name)
    uploaded_stem = _filename_stem(file_name)

    if not uploaded_name:
        return None, None, None

    # -------------------------------------------------------------------------
    # FIRST PASS:
    # Exact filename / basename matching.
    # -------------------------------------------------------------------------

    for _, row in df.iterrows():

        csv_value = row.get(image_column, "")

        csv_full = _normalize_text(csv_value)
        csv_name = _normalize_filename(csv_value)
        csv_stem = _filename_stem(csv_value)

        # Exact full path or exact filename.
        exact_match = (
            uploaded_full == csv_full
            or uploaded_name == csv_name
        )

        if exact_match:

            try:
                lat = float(row[latitude_column])
                lon = float(row[longitude_column])

                if pd.isna(lat) or pd.isna(lon):
                    continue

                if event_column is not None:
                    event = str(row.get(
                        event_column,
                        "CSV Dataset Event"
                    ))
                else:
                    event = "CSV Dataset Event"

                return (
                    lat,
                    lon,
                    event
                )

            except (ValueError, TypeError, KeyError):
                continue

    # -------------------------------------------------------------------------
    # SECOND PASS:
    # Match filename without extension.
    #
    # Example:
    #
    # uploaded:
    #     d3.jpeg
    #
    # CSV:
    #     d3.jpg
    #
    # Both become:
    #     d3
    #
    # This handles common dataset extension differences.
    # -------------------------------------------------------------------------

    if uploaded_stem:

        for _, row in df.iterrows():

            csv_value = row.get(image_column, "")

            csv_stem = _filename_stem(csv_value)

            if (
                csv_stem
                and uploaded_stem == csv_stem
            ):

                try:

                    lat = float(row[latitude_column])
                    lon = float(row[longitude_column])

                    if pd.isna(lat) or pd.isna(lon):
                        continue

                    if event_column is not None:
                        event = str(row.get(
                            event_column,
                            "CSV Dataset Event"
                        ))
                    else:
                        event = "CSV Dataset Event"

                    return (
                        lat,
                        lon,
                        event
                    )

                except (ValueError, TypeError, KeyError):
                    continue

    # -------------------------------------------------------------------------
    # No match.
    # -------------------------------------------------------------------------

    return None, None, None


# =============================================================================
# OPTIONAL DEBUG FUNCTION
# =============================================================================

def describe_csv_columns(df_metadata):
    """
    Useful for debugging during the hackathon.

    Returns the detected column mapping.
    """

    if df_metadata is None or df_metadata.empty:
        return {}

    df = df_metadata.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return {
        "image_column": _find_column(
            df,
            [
                "Image_Path",
                "Image Path",
                "Filename",
                "File_Name",
                "Image",
                "Path",
            ]
        ),
        "latitude_column": _find_column(
            df,
            [
                "Latitude",
                "Lat",
                "GPS_Latitude",
            ]
        ),
        "longitude_column": _find_column(
            df,
            [
                "Longitude",
                "Lon",
                "Long",
                "GPS_Longitude",
            ]
        ),
        "event_column": _find_column(
            df,
            [
                "Event",
                "Disaster",
                "Event_Name",
            ]
        ),
    }