import os
import streamlit as st
import pandas as pd

from backend.database import save_db
from backend.csv_registry import lookup_coordinates_from_csv
from backend.ingestion import (
    create_image_upload,
    create_video_upload,
    get_location_for_file,
)
from backend.image_ai import run_accurate_ai_inference
from backend.video_ai import analyze_video
from frontend.map_view import render_original_map


IMAGE_EXTS = (".jpg", ".png", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov")


# =============================================================================
# ANALYZE PENDING INFRASTRUCTURE FEEDS
# =============================================================================

def _analyze_pending(db_data):

    pending = [
        item
        for item in db_data.get("public_uploads", [])
        if item.get("verified")
        and not item.get("analyzed")
        and "Infra" in item.get("source", "")
    ]

    if not pending:
        st.info(
            "All verified infrastructure feeds have already been analyzed!"
        )
        return

    with st.spinner(
        f"Running AI Vision Engine on {len(pending)} pending infrastructure feeds..."
    ):

        for item in db_data["public_uploads"]:

            if (
                not item.get("verified")
                or item.get("analyzed")
                or "Infra" not in item.get("source", "")
            ):
                continue

            # -------------------------------------------------------------
            # VIDEO
            # -------------------------------------------------------------

            if item.get("media_type") == "video":

                result = analyze_video(
                    item["file_path"]
                )

                item["category"] = result["category"]

                item["severity"] = result["severity"]

                item["annotated_path"] = (
                    result["annotated_path"]
                    or item["file_path"]
                )

                item["video_frames_analyzed"] = (
                    result["frames_analyzed"]
                )

                item["video_frame_results"] = (
                    result["frame_results"]
                )

            # -------------------------------------------------------------
            # IMAGE
            # -------------------------------------------------------------

            else:

                cat, sev, anno_p = (
                    run_accurate_ai_inference(
                        item["file_path"]
                    )
                )

                item["category"] = cat
                item["severity"] = sev
                item["annotated_path"] = anno_p

            # -------------------------------------------------------------
            # COLOR CLASSIFICATION
            # -------------------------------------------------------------

            sev = item.get("severity", 0)

            item["color"] = (
                "RED"
                if sev >= 70
                else (
                    "YELLOW"
                    if sev >= 40
                    else "GREEN"
                )
            )

            item["analyzed"] = True

        save_db(db_data)

        st.success(
            f"Processed {len(pending)} infrastructure feeds successfully!"
        )

        st.rerun()


# =============================================================================
# INFRASTRUCTURE TAB
# =============================================================================

def render_infrastructure_tab(db_data, drone_info):

    st.header(
        "Infrastructure Assessment & Multi-Sensor Feeds"
    )

    st.write(
        "Upload drone videos/photos, satellite imagery, CCTV feeds, "
        "or ingest folder datasets with embedded CSVs."
    )

    # =========================================================================
    # TWO COLUMN LAYOUT
    # =========================================================================

    col_left, col_right = st.columns([1, 1])


    # =========================================================================
    # LEFT COLUMN
    # =========================================================================

    with col_left:

        st.subheader(
            "1. Dedicated Multi-Source Ingestion"
        )

        infra_source = st.selectbox(
            "Select Data Source",
            [
                "Drone Video / Photos",
                "Satellite Imagery",
                "CCTV Stream / File",
                "Bulk Dataset Folder (Co-existing CSV)"
            ],
            key="infra_source_select"
        )


        # =====================================================================
        # BULK DATASET
        # =====================================================================

        if infra_source == "Bulk Dataset Folder (Co-existing CSV)":

            st.info(
                "Upload multiple images and CSV files from your "
                "local infrastructure folder together."
            )

            bulk_files = st.file_uploader(
                "Select Folder Files (Upload Images & CSVs Together)",
                accept_multiple_files=True,
                key="infra_bulk_uploader"
            )

            if st.button(
                "Ingest Dataset Folder",
                type="primary",
                use_container_width=True,
                key="infra_bulk_btn"
            ):

                if bulk_files:

                    folder_csvs = [
                        f
                        for f in bulk_files
                        if f.name.lower().endswith(".csv")
                    ]

                    df_folder = None


                    # ---------------------------------------------------------
                    # READ INFRASTRUCTURE CSV
                    # ---------------------------------------------------------

                    if folder_csvs:

                        try:

                            parsed_dfs = [
                                pd.read_csv(c)
                                for c in folder_csvs
                            ]

                            for d in parsed_dfs:
                                d.columns = (
                                    d.columns
                                    .str.strip()
                                )

                            df_folder = pd.concat(
                                parsed_dfs,
                                ignore_index=True
                            )

                            st.session_state.infra_csv_metadata = (
                                df_folder
                            )

                            st.success(
                                f"Parsed {len(folder_csvs)} embedded "
                                f"folder CSVs "
                                f"({len(df_folder)} records)!"
                            )

                        except Exception as e:

                            st.error(
                                f"Error parsing folder CSVs: {e}"
                            )

                    elif (
                        st.session_state.get(
                            "infra_csv_metadata"
                        ) is not None
                    ):

                        df_folder = (
                            st.session_state.infra_csv_metadata
                        )


                    # ---------------------------------------------------------
                    # INGEST IMAGES
                    # ---------------------------------------------------------

                    ingested_count = 0

                    for f in bulk_files:

                        if f.name.lower().endswith(
                            IMAGE_EXTS
                        ):

                            lat_val, lon_val, event_val = (
                                get_location_for_file(
                                    f.name,
                                    df_folder,
                                    "Bulk Dataset Ingest"
                                )
                            )

                            create_image_upload(
                                db_data,
                                f,
                                "Infra: Bulk Dataset",
                                lat_val,
                                lon_val,
                                event_val,
                                drone_info
                            )

                            ingested_count += 1


                    save_db(db_data)

                    st.success(
                        f"Successfully ingested "
                        f"{ingested_count} infrastructure images!"
                    )

                    st.rerun()


        # =====================================================================
        # SINGLE SOURCE UPLOAD
        # =====================================================================

        else:

            infra_file = st.file_uploader(
                f"Upload {infra_source}",
                type=[
                    "jpg",
                    "png",
                    "jpeg",
                    "webp",
                    "mp4",
                    "avi",
                    "mov"
                ],
                key="infra_single_uploader"
            )


            # -----------------------------------------------------------------
            # CSV LOCATION LOOKUP
            # -----------------------------------------------------------------

            if infra_file is not None:

                if (
                    st.session_state.get(
                        "last_processed_infra_file"
                    )
                    != infra_file.name
                ):

                    st.session_state.last_processed_infra_file = (
                        infra_file.name
                    )


                    lat_val, lon_val, event_val = (
                        lookup_coordinates_from_csv(
                            infra_file.name,
                            st.session_state.get(
                                "infra_csv_metadata"
                            )
                        )
                    )


                    # ---------------------------------------------------------
                    # CSV MATCH FOUND
                    # ---------------------------------------------------------

                    if (
                        lat_val is not None
                        and lon_val is not None
                    ):

                        st.session_state[
                            "infra_lat_in"
                        ] = float(lat_val)

                        st.session_state[
                            "infra_lon_in"
                        ] = float(lon_val)

                        st.session_state[
                            "infra_event_val"
                        ] = event_val

                        st.success(
                            f"Matched Infra CSV coordinates: "
                            f"`{lat_val}, {lon_val}`"
                        )


                    # ---------------------------------------------------------
                    # CSV MATCH NOT FOUND
                    # ---------------------------------------------------------

                    else:

                        if (
                            "nepal"
                            in infra_file.name.lower()
                        ):

                            st.session_state[
                                "infra_lat_in"
                            ] = 27.7172

                            st.session_state[
                                "infra_lon_in"
                            ] = 85.3240

                            st.session_state[
                                "infra_event_val"
                            ] = "Nepal Floods"

                        else:

                            st.session_state[
                                "infra_lat_in"
                            ] = 0.0

                            st.session_state[
                                "infra_lon_in"
                            ] = 0.0

                            st.session_state[
                                "infra_event_val"
                            ] = (
                                f"Infra: {infra_source}"
                            )

                            st.warning(
                                "Coordinates not found in "
                                "Infrastructure CSV. "
                                "You can enter them manually."
                            )


            # -----------------------------------------------------------------
            # LATITUDE
            # -----------------------------------------------------------------

            i_lat = st.number_input(
                "Latitude",
                format="%.4f",
                key="infra_lat_in"
            )


            # -----------------------------------------------------------------
            # LONGITUDE
            # -----------------------------------------------------------------

            i_lon = st.number_input(
                "Longitude",
                format="%.4f",
                key="infra_lon_in"
            )


            # -----------------------------------------------------------------
            # EVENT
            # -----------------------------------------------------------------

            event_tag = st.session_state.get(
                "infra_event_val",
                f"Infra: {infra_source}"
            )


            # -----------------------------------------------------------------
            # SUBMIT
            # -----------------------------------------------------------------

            if st.button(
                "Submit Infrastructure Feed",
                type="primary",
                use_container_width=True,
                key="infra_sub_btn"
            ):

                if infra_file is not None:

                    # ---------------------------------------------------------
                    # VIDEO
                    # ---------------------------------------------------------

                    if infra_file.name.lower().endswith(
                        VIDEO_EXTS
                    ):

                        create_video_upload(
                            db_data,
                            infra_file,
                            f"Infra: {infra_source}",
                            i_lat,
                            i_lon,
                            event_tag,
                            drone_info
                        )

                        st.info(
                            "Video feed recorded! Frame processing "
                            "will be initiated during AI analysis."
                        )


                    # ---------------------------------------------------------
                    # IMAGE
                    # ---------------------------------------------------------

                    else:

                        create_image_upload(
                            db_data,
                            infra_file,
                            f"Infra: {infra_source}",
                            i_lat,
                            i_lon,
                            event_tag,
                            drone_info
                        )


                    save_db(db_data)

                    st.success(
                        "Infrastructure media registered successfully!"
                    )

                    st.rerun()


    # =========================================================================
    # RIGHT COLUMN
    # =========================================================================

    with col_right:

        st.subheader(
            "2. Infrastructure AI Analysis & Stream"
        )

        uploads_list = [
            item
            for item in db_data.get(
                "public_uploads",
                []
            )
            if "Infra" in item.get(
                "source",
                ""
            )
        ]


        # ---------------------------------------------------------------------
        # PENDING COUNT
        # ---------------------------------------------------------------------

        pending_count = sum(
            1
            for item in uploads_list
            if item.get("verified")
            and not item.get("analyzed")
        )

        st.markdown(
            f"**Pending Unanalyzed Infrastructure Feeds:** "
            f"`{pending_count}`"
        )


        # ---------------------------------------------------------------------
        # BUTTONS
        # ---------------------------------------------------------------------

        col_a2, col_r2 = st.columns(2)

        with col_a2:

            run_infra_ai = st.button(
                "Refresh & Analyze All Pending Photos",
                type="primary",
                use_container_width=True,
                key="infra_ai_run"
            )

        with col_r2:

            if st.button(
                "Sync Feed / Refresh View",
                use_container_width=True,
                key="infra_sync_btn"
            ):

                st.rerun()


        if run_infra_ai:

            _analyze_pending(db_data)


        st.divider()


        # =====================================================================
        # HIDDEN UPLOAD HISTORY
        # =====================================================================
        #
        # IMPORTANT:
        # The previous version displayed every upload directly.
        #
        # Now the complete history is hidden inside ONE expander.
        #
        # User clicks:
        #
        #   📂 View Infrastructure Upload History
        #
        # and only then the individual feeds appear.
        #
        # =====================================================================

        with st.expander(
            f"📂 View Infrastructure Upload History "
            f"({len(uploads_list)} feeds)",
            expanded=False
        ):

            if not uploads_list:

                st.info(
                    "No infrastructure feeds uploaded yet."
                )

            else:

                for idx, item in enumerate(
                    reversed(uploads_list)
                ):

                    with st.expander(
                        f"[{item.get('source', 'Infra')}] "
                        f"{item.get('filename', 'Unknown')} | "
                        f"{item.get('timestamp', '')}",
                        expanded=False
                    ):

                        # -----------------------------------------------------
                        # MEDIA
                        # -----------------------------------------------------

                        with st.expander(
                            "📁 View Uploaded Media / Annotated Output",
                            expanded=False
                        ):

                            if (
                                item.get("media_type")
                                == "video"
                            ):

                                st.video(
                                    item["file_path"]
                                )

                            else:

                                image_path = (
                                    item.get(
                                        "annotated_path"
                                    )
                                    if item.get(
                                        "analyzed"
                                    )
                                    else item.get(
                                        "file_path"
                                    )
                                )

                                if (
                                    image_path
                                    and os.path.exists(
                                        image_path
                                    )
                                ):

                                    st.image(
                                        image_path,
                                        use_container_width=True
                                    )

                                else:

                                    st.warning(
                                        "Image file is not available."
                                    )


                        # -----------------------------------------------------
                        # LOCATION
                        # -----------------------------------------------------

                        st.caption(
                            f"**Coordinates:** "
                            f"`{item.get('lat', 0)}, "
                            f"{item.get('lon', 0)}` | "
                            f"**Event:** "
                            f"`{item.get('event', 'N/A')}`"
                        )


                        # -----------------------------------------------------
                        # AI OUTPUT
                        # -----------------------------------------------------

                        if item.get("analyzed"):

                            st.write(
                                f"**Type:** "
                                f"`{item.get('category', 'Infrastructure')}`"
                            )

                            st.write(
                                f"**Severity Score:** "
                                f"`{item.get('severity', 0)}%`"
                            )

                            sev = item.get(
                                "severity",
                                0
                            )

                            if sev >= 70:

                                st.error(
                                    "🔴 HIGH RISK (RED)"
                                )

                            elif sev >= 40:

                                st.warning(
                                    "🟡 MODERATE RISK (YELLOW)"
                                )

                            else:

                                st.success(
                                    "🟢 LOW RISK (GREEN)"
                                )

                        else:

                            st.info(
                                "Pending Analysis — click "
                                "'Refresh & Analyze All Pending Photos' "
                                "above."
                            )


    # =========================================================================
    # INFRASTRUCTURE DAMAGE & INSPECTION INTELLIGENCE
    # =========================================================================

    st.divider()

    st.header(
        "Infrastructure Damage & Inspection Intelligence"
    )


    # Only infrastructure items
    analyzed_infra_items = [
        item
        for item in db_data.get(
            "public_uploads",
            []
        )
        if (
            item.get("analyzed", False)
            and "Infra" in item.get(
                "source",
                ""
            )
        )
    ]


    if not analyzed_infra_items:

        st.info(
            "No analyzed infrastructure media available. "
            "Run 'Refresh & Analyze All Pending Photos' above "
            "to populate damage assessment outputs."
        )


    else:

        # =====================================================================
        # DETAILED DAMAGE ASSESSMENT
        # =====================================================================

        st.subheader(
            "1. Detailed Infrastructure Damage Assessment"
        )

        table_data = []

        for idx, item in enumerate(
            analyzed_infra_items,
            1
        ):

            table_data.append(
                {
                    "ID": f"INF-{idx:03d}",

                    "Asset / File": item.get(
                        "filename",
                        "Unknown"
                    ),

                    "Damage Location": (
                        f"{float(item.get('lat', 0)):.4f}, "
                        f"{float(item.get('lon', 0)):.4f} "
                        f"({item.get('event', 'Zone')})"
                    ),

                    "Structure Category": item.get(
                        "category",
                        "Structure"
                    ),

                    "Estimated Severity": (
                        f"{item.get('severity', 0)}%"
                    ),

                    "Source Media": item.get(
                        "source",
                        "Upload"
                    )
                }
            )


        df_assessment = pd.DataFrame(
            table_data
        )

        st.dataframe(
            df_assessment,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================================
        # PRIORITIZED EMERGENCY INSPECTION LIST
        # =====================================================================

        st.divider()

        st.subheader(
            "2. Prioritized Emergency Inspection List"
        )

        st.caption(
            "Ranked response order generated for deployment teams. "
            "Each item includes a visual glimpse of the analyzed image."
        )


        prioritized_items = sorted(
            analyzed_infra_items,
            key=lambda x: x.get(
                "severity",
                0
            ),
            reverse=True
        )


        with st.container(
            height=520,
            border=True
        ):

            for rank, item in enumerate(
                prioritized_items,
                1
            ):

                sev = item.get(
                    "severity",
                    0
                )

                asset_label = (
                    f"{item.get('category', 'Structure')} "
                    f"- {item.get('filename', 'Unknown')}"
                )


                # -------------------------------------------------------------
                # PRIORITY
                # -------------------------------------------------------------

                if sev >= 70:

                    badge = "🔴 CRITICAL PRIORITY"

                    why = (
                        f"Severe visible damage "
                        f"({sev}%) in "
                        f"{item.get('event', 'the area')}."
                    )

                    rec_action = (
                        "Send rapid structural assessment team "
                        "immediately and erect barriers."
                    )

                elif sev >= 40:

                    badge = "🟡 HIGH PRIORITY"

                    why = (
                        f"Moderate structural/road obstruction "
                        f"({sev}%)."
                    )

                    rec_action = (
                        "Deploy clearance crew and perform "
                        "secondary structural verification."
                    )

                else:

                    badge = "🟢 LOW PRIORITY"

                    why = (
                        f"Minor visible damage "
                        f"({sev}%)."
                    )

                    rec_action = (
                        "Log incident and schedule standard "
                        "inspection during routine maintenance."
                    )


                # -------------------------------------------------------------
                # PRIORITY HEADER
                # -------------------------------------------------------------

                st.markdown(
                    f"#### Inspection Priority #{rank}: "
                    f"{asset_label}"
                )

                st.markdown(
                    f"**Status:** `{badge}` | "
                    f"**Severity:** `{sev}%` | "
                    f"**Location:** "
                    f"`{float(item.get('lat', 0)):.4f}, "
                    f"{float(item.get('lon', 0)):.4f}`"
                )


                # -------------------------------------------------------------
                # IMAGE GLIMPSE
                # -------------------------------------------------------------

                image_path = (
                    item.get("annotated_path")
                    if item.get("analyzed")
                    else item.get("file_path")
                )

                if (
                    image_path
                    and os.path.exists(image_path)
                ):

                    with st.expander(
                        "🖼️ View Damage Image",
                        expanded=False
                    ):

                        st.image(
                            image_path,
                            use_container_width=True
                        )


                # -------------------------------------------------------------
                # WHY
                # -------------------------------------------------------------

                st.markdown(
                    f"**Why:** {why}"
                )


                # -------------------------------------------------------------
                # RECOMMENDED ACTION
                # -------------------------------------------------------------

                st.markdown(
                    f"**Recommended Action:** "
                    f"{rec_action}"
                )


                # -------------------------------------------------------------
                # GOOGLE MAPS NAVIGATION
                # -------------------------------------------------------------

                lat = item.get(
                    "lat",
                    0
                )

                lon = item.get(
                    "lon",
                    0
                )

                nav_url = (
                    "https://www.google.com/maps/dir/"
                    "?api=1"
                    f"&destination={lat},{lon}"
                )

                st.markdown(
                    f"👉 **[Launch Navigation Route "
                    f"to Location]({nav_url})**"
                )

                st.divider()


    # =========================================================================
    # SHARED MAP
    # =========================================================================

    st.divider()

    render_original_map(
        db_data.get(
            "public_uploads",
            []
        ),
        drone_info,
        "infra"
    )