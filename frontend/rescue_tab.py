import streamlit as st
from collections import Counter

from frontend.map_view import render_original_map


def render_rescue_tab(db_data, drone_info):
    st.header("Rescue Operations & Priority Dispatch")
    st.write(
        "Overview of validated disaster locations categorized by "
        "damage severity for response teams."
    )

    uploads_list = db_data.get("public_uploads", [])
    sos_list = db_data.get("sos_signals", [])

    analyzed_items = [
        i for i in uploads_list
        if i.get("analyzed", False)
    ]

    if not analyzed_items:
        st.info(
            "No analyzed disaster locations available yet. "
            "Run AI Analysis in Public or Infrastructure tabs."
        )
    else:
        high_risk = [
            i for i in analyzed_items
            if i["color"] == "RED"
        ]
        mod_risk = [
            i for i in analyzed_items
            if i["color"] == "YELLOW"
        ]
        low_risk = [
            i for i in analyzed_items
            if i["color"] == "GREEN"
        ]
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric(
            "High Severity (RED)",
            len(high_risk)
        )
        col_r2.metric(
            "Moderate Risk (YELLOW)",
            len(mod_risk)
        )
        col_r3.metric(
            "Safe / Intact (GREEN)",
            len(low_risk)
        )

    st.divider()

    # =========================================================================
    # SOS SIGNAL INTELLIGENCE
    # =========================================================================

    st.subheader("🆘 SOS Signal Intelligence")

    if not sos_list:

        st.info("No SOS signals received yet.")

    else:

        st.metric(
            "Total SOS Signals Received",
            len(sos_list)
        )

        # ---------------------------------------------------------------
        # Simple grid-based hotspot clustering.
        # Rounding to 2 decimal places groups points within roughly
        # a ~1.1 km x 1.1 km cell — good enough to spot "where signals
        # are coming from more" without needing a real clustering lib.
        # ---------------------------------------------------------------

        precision = 2

        cluster_counts = Counter()
        cluster_points = {}

        for sig in sos_list:

            try:
                c_lat = round(float(sig["lat"]), precision)
                c_lon = round(float(sig["lon"]), precision)
            except (KeyError, TypeError, ValueError):
                continue

            key = (c_lat, c_lon)

            cluster_counts[key] += 1
            cluster_points.setdefault(key, []).append(sig)

        ranked_clusters = cluster_counts.most_common()

        if ranked_clusters:

            top_key, top_count = ranked_clusters[0]

            st.warning(
                f"📍 Highest concentration of SOS signals: "
                f"**{top_count} signal(s)** near "
                f"`{top_key[0]}, {top_key[1]}`"
            )

            st.markdown("#### Prioritized SOS Response Order")

            for rank, (key, count) in enumerate(ranked_clusters, start=1):

                signals_here = cluster_points[key]

                latest = max(
                    signals_here,
                    key=lambda s: s.get("timestamp", "")
                )

                nav_url = (
                    "https://www.google.com/maps/dir/?api=1"
                    f"&destination={key[0]},{key[1]}"
                )

                with st.container(border=True):

                    st.markdown(
                        f"**Priority #{rank} — {count} signal(s) "
                        f"in this area**"
                    )

                    st.caption(
                        f"Approx. Location: `{key[0]}, {key[1]}` | "
                        f"Most Recent: {latest.get('timestamp', 'N/A')}"
                    )

                    st.markdown(
                        f"[📍 Launch Navigation Route]({nav_url})"
                    )

    st.divider()

    render_original_map(
        db_data.get("public_uploads", []),
        drone_info,
        "rescue",
        sos_list=sos_list
    )