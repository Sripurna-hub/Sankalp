import streamlit as st

def render_home():
    st.title("Autonomous Disaster Response Engine")
    st.subheader("Select Platform Operating Mode")

    col1, col2 = st.columns(2)

    with col1:
        st.info("### Hackathon Demo Version")
        st.write(
            "Access interactive public upload streams, drone geofence "
            "verification, AI analysis, and GIS routing."
        )
        if st.button(
            "Launch Hackathon Demo Mode",
            use_container_width=True
        ):
            st.session_state.app_mode = "demo"
            st.rerun()

    with col2:
        st.error("### Active Disaster Event Mode")
        st.write(
            "Locked mode for emergency services during live operations "
            "with automated drone telemetry integration."
        )
        if st.button(
            "Launch Active Event Mode",
            use_container_width=True
        ):
            st.session_state.app_mode = "active"
            st.rerun()
