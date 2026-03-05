import os
import streamlit as st

from services.resume_services import (
    extract_text_from_resume,
    analyze_resume,
    extract_suitability_score,
    store_resume_analysis
)


def main():
    st.set_page_config(
        page_title="Resume Screening AI",
        layout="wide"
    )
    st.title("Resume Screening with AI")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Job Requirements")
        job_requirements = st.text_area(
            "Enter Job Requirements",
            height=300
        )

    with col2:
        st.header("Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload Resume",
            type=["pdf", "docx", "txt"]
        )

    if st.button("Analyze"):

        if not uploaded_file or not job_requirements:
            st.warning("Please upload a resume and enter job requirements")
            return

        with st.spinner("Processing Resume..."):

            resume_text = extract_text_from_resume(uploaded_file)

            with st.expander("View Resume Text"):
                st.write(resume_text)

            analysis = analyze_resume(job_requirements, resume_text)

            st.header("AI Analysis")
            st.markdown(analysis)

            suitability_score = extract_suitability_score(analysis)

            if suitability_score is not None:
                st.metric(
                    label="Resume Suitability Score",
                    value=f"{suitability_score}%"
                )

            store_resume_analysis(
                analysis,
                os.path.splitext(uploaded_file.name)[0]
            )

            st.success("Analysis stored in vector database")

            st.download_button(
                "Download Analysis",
                analysis,
                file_name="resume_analysis.txt"
            )


if __name__ == "__main__":
    main()