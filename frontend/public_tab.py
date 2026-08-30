import os
import datetime
import streamlit as st

from backend.database import save_db
from backend.csv_registry import lookup_coordinates_from_csv
from backend.ingestion import create_image_upload
from backend.image_ai import run_accurate_ai_inference
from frontend.map_view import render_original_map


# =============================================================================
# SOS SIGNAL CREATION
# =============================================================================

def _create_sos_signal(db_data, lat, lon):
    sos_list = db_data.setdefault("sos_signals", [])

    sos_id = f"SOS-{len(sos_list) + 1:03d}"

    sos_list.append({
        "id": sos_id,
        "lat": float(lat),
        "lon": float(lon),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active"
    })

    return sos_id


# =============================================================================
# PROCESS PENDING PUBLIC IMAGES
# =============================================================================

def _process_pending_images(db_data):

    pending = [
        item
        for item in db_data.get("public_uploads", [])
        if item.get("verified", False)
        and not item.get("analyzed", False)
        and item.get("media_type", "image") == "image"
    ]

    if not pending:

        st.info(
            "All verified uploaded photos have already been analyzed!"
        )

        return


    with st.spinner(
        f"Running Local AI Engine on "
        f"{len(pending)} pending uploads..."
    ):

        for item in db_data["public_uploads"]:

            if (
                item.get("verified", False)
                and not item.get("analyzed", False)
                and item.get(
                    "media_type",
                    "image"
                ) == "image"
            ):

                cat, sev, anno_p = (
                    run_accurate_ai_inference(
                        item["file_path"]
                    )
                )


                color = (
                    "RED"
                    if sev >= 70
                    else (
                        "YELLOW"
                        if sev >= 40
                        else "GREEN"
                    )
                )


                item["analyzed"] = True
                item["category"] = cat
                item["severity"] = sev
                item["color"] = color
                item["annotated_path"] = anno_p


        save_db(db_data)

        st.success(
            f"Successfully processed "
            f"{len(pending)} photos with Local AI!"
        )

        st.rerun()


# =============================================================================
# PUBLIC TAB
# =============================================================================

def render_public_tab(
    db_data,
    drone_info
):

    st.header(
        "Citizen Multi-Source Upload & Verification"
    )


    # =========================================================================
    # EMERGENCY SOS
    # =========================================================================

    st.subheader("🆘 Emergency SOS")

    st.caption(
        "If you are in immediate danger, send an SOS signal. "
        "This alerts rescue teams to your location and can only "
        "be sent once."
    )

    sos_already_sent = st.session_state.get(
        "sos_sent_flag",
        False
    )

    if sos_already_sent:

        st.error(
            "🆘 SOS Sent — rescue teams have been alerted "
            "to your location."
        )

    else:

        sos_c1, sos_c2 = st.columns(2)

        with sos_c1:

            sos_lat = st.number_input(
                "Your Lat Coordinate",
                format="%.4f",
                value=float(drone_info["lat"]),
                key="sos_lat_in"
            )

        with sos_c2:

            sos_lon = st.number_input(
                "Your Lon Coordinate",
                format="%.4f",
                value=float(drone_info["lon"]),
                key="sos_lon_in"
            )

        send_sos = st.button(
            "🆘 SEND SOS",
            type="primary",
            use_container_width=True,
            key="sos_send_btn"
        )

        if send_sos:

            _create_sos_signal(
                db_data,
                sos_lat,
                sos_lon
            )

            save_db(db_data)

            st.session_state["sos_sent_flag"] = True

            st.success(
                "SOS sent successfully! Rescue teams have "
                "been notified."
            )

            st.rerun()


    st.divider()


    c_upload, c_feed = st.columns([1, 1])


    # =========================================================================
    # CITIZEN UPLOAD
    # =========================================================================

    with c_upload:

        st.subheader(
            "1. Citizen Photo Ingestion"
        )


        uploaded_photo = st.file_uploader(
            "Upload Damage Photo",
            type=[
                "jpg",
                "png",
                "jpeg",
                "webp"
            ],
            key="cit_photo_upload"
        )


        if uploaded_photo is not None:

            current_file_id = (
                f"{uploaded_photo.name}_"
                f"{getattr(uploaded_photo, 'size', 0)}"
            )

            previous_file_id = (
                st.session_state.get(
                    "last_processed_cit_file_id"
                )
            )


            # -------------------------------------------------------------
            # LOCATION LOOKUP
            # -------------------------------------------------------------

            if current_file_id != previous_file_id:

                st.session_state.last_processed_cit_file_id = (
                    current_file_id
                )


                (
                    lat_val,
                    lon_val,
                    event_val
                ) = lookup_coordinates_from_csv(
                    uploaded_photo.name,
                    st.session_state.get(
                        "public_csv_metadata"
                    )
                )


                # ---------------------------------------------------------
                # CSV MATCH
                # ---------------------------------------------------------

                if (
                    lat_val is not None
                    and lon_val is not None
                ):

                    st.session_state["cit_lat_in"] = (
                        float(lat_val)
                    )

                    st.session_state["cit_lon_in"] = (
                        float(lon_val)
                    )

                    st.session_state["cit_event_val"] = (
                        event_val
                        or "CSV Verified Event"
                    )

                    st.success(
                        f"Matched CSV coordinates: "
                        f"`{float(lat_val):.4f}, "
                        f"{float(lon_val):.4f}` "
                        f"({event_val})"
                    )


                # ---------------------------------------------------------
                # FALLBACK
                # ---------------------------------------------------------

                else:

                    if "nepal" in uploaded_photo.name.lower():

                        st.session_state["cit_lat_in"] = (
                            27.7172
                        )

                        st.session_state["cit_lon_in"] = (
                            85.3240
                        )

                        st.session_state["cit_event_val"] = (
                            "Nepal Floods"
                        )

                    else:

                        st.session_state["cit_lat_in"] = (
                            0.0
                        )

                        st.session_state["cit_lon_in"] = (
                            0.0
                        )

                        st.session_state["cit_event_val"] = (
                            "Manual Upload"
                        )

                    st.warning(
                        "Coordinates not found in CSV. "
                        "Applied location fallback."
                    )


        # ---------------------------------------------------------------------
        # COORDINATES
        # ---------------------------------------------------------------------

        input_lat = st.number_input(
            "Lat Coordinate",
            format="%.4f",
            key="cit_lat_in"
        )


        input_lon = st.number_input(
            "Lon Coordinate",
            format="%.4f",
            key="cit_lon_in"
        )


        matched_event = st.session_state.get(
            "cit_event_val",
            "Manual Upload"
        )


        # ---------------------------------------------------------------------
        # SUBMIT
        # ---------------------------------------------------------------------

        submit_upload = st.button(
            "Submit Photo",
            type="primary",
            use_container_width=True,
            key="cit_submit_btn"
        )


        if (
            submit_upload
            and uploaded_photo is not None
        ):

            create_image_upload(
                db_data,
                uploaded_photo,
                "Citizen Upload",
                input_lat,
                input_lon,
                matched_event,
                drone_info
            )

            save_db(db_data)

            st.success(
                "Uploaded successfully! Synced across network."
            )

            st.rerun()


    # =========================================================================
    # PUBLIC FEED
    # =========================================================================

    with c_feed:

        st.subheader(
            "2. Real-Time Shared Upload Stream"
        )


        uploads_list = db_data.get(
            "public_uploads",
            []
        )


        pending_count = sum(
            1
            for item in uploads_list
            if item.get("verified", False)
            and not item.get("analyzed", False)
            and item.get(
                "media_type",
                "image"
            ) == "image"
        )


        st.markdown(
            f"**Pending Unanalyzed Images in Database:** "
            f"`{pending_count}`"
        )


        col_a, col_r = st.columns(2)


        with col_a:

            run_batch_ai = st.button(
                "Refresh & Analyze All Pending Photos",
                type="primary",
                use_container_width=True,
                key="cit_analyze_btn"
            )


        with col_r:

            if st.button(
                "Sync Feed / Refresh View",
                use_container_width=True,
                key="cit_sync_btn"
            ):

                st.rerun()


        if run_batch_ai:

            _process_pending_images(
                db_data
            )


        st.divider()


        # =====================================================================
        # COLLAPSED PUBLIC INPUT HISTORY
        # =====================================================================

        if not uploads_list:

            st.info(
                "No uploads yet. Upload media to begin."
            )

        else:

            with st.expander(
                f"📂 Upload History "
                f"({len(uploads_list)} feeds)",
                expanded=False
            ):

                for idx, item in enumerate(
                    reversed(uploads_list)
                ):

                    with st.expander(
                        f"[{item.get('source', 'Citizen')}] "
                        f"{item['filename']} | "
                        f"{item['timestamp']}",
                        expanded=False
                    ):

                        # -----------------------------------------------------
                        # MEDIA
                        # -----------------------------------------------------

                        with st.expander(
                            "📁 View Uploaded Media / "
                            "Annotated Output",
                            expanded=False
                        ):

                            media_path = (
                                item.get("annotated_path")
                                if item.get("analyzed")
                                else item.get("file_path")
                            )


                            if (
                                item.get("media_type")
                                == "video"
                            ):

                                if (
                                    item.get("file_path")
                                    and os.path.exists(
                                        item["file_path"]
                                    )
                                ):

                                    st.video(
                                        item["file_path"]
                                    )

                            else:

                                if (
                                    media_path
                                    and os.path.exists(
                                        media_path
                                    )
                                ):

                                    st.image(
                                        media_path,
                                        use_container_width=True
                                    )


                        # -----------------------------------------------------
                        # VERIFICATION
                        # -----------------------------------------------------

                        if item.get(
                            "verified",
                            False
                        ):

                            st.success(
                                "Drone Verification: "
                                f"{item.get('status_reason', 'VERIFIED')}"
                            )

                        else:

                            st.error(
                                "Drone Verification: "
                                f"{item.get('status_reason', 'NOT VERIFIED')}"
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

                        if item.get("analyzed", False):

                            st.markdown(
                                "#### AI Assessment Output"
                            )


                            st.write(
                                f"**Infrastructure Class:** "
                                f"`{item.get('category', 'N/A')}`"
                            )


                            st.write(
                                f"**Damage Severity Score:** "
                                f"`{item.get('severity', 0)}%`"
                            )


                            color = item.get(
                                "color",
                                "GREEN"
                            )


                            if color == "RED":

                                st.error(
                                    "Tag: HIGH RISK (RED)"
                                )

                            elif color == "YELLOW":

                                st.warning(
                                    "Tag: MODERATE RISK (YELLOW)"
                                )

                            else:

                                st.success(
                                    "Tag: LOW RISK / SAFE (GREEN)"
                                )


                            # -------------------------------------------------
                            # NAVIGATION
                            # -------------------------------------------------

                            nav_url = (
                                "https://www.google.com/maps/dir/?api=1"
                                f"&destination="
                                f"{item.get('lat', 0)},"
                                f"{item.get('lon', 0)}"
                            )


                            st.markdown(
                                f"**[Open Shortest Navigation Path "
                                f"in Google Maps]({nav_url})**"
                            )


                        else:

                            st.info(
                                "Pending Analysis "
                                "(Click 'Refresh & Analyze All "
                                "Pending Photos' above)"
                            )


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
        "public"
    )