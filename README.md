## RAG-Based Resume and Job Description Matching

This repository contains a Retrieval-Augmented Generation (RAG) pipeline that evaluates a list of resumes based on a given job description. The pipeline utilizes Named Entity Recognition (NER) for skill recognition and RAG-based Large Language Model (LLM) prompting to assess the proficiency of skills in each resume. The system then provides a score for each skill based on how well it matches the job requirements.

### Overview

This pipeline automates the process of evaluating resumes for a given job description, with the primary goal of assessing the proficiency of skills mentioned in the resumes. By combining NER and RAG techniques, the pipeline provides an efficient way to match resumes with job descriptions and rank candidates according to their skill proficiency.

### Features

Skill Recognition with NER: Automatically extracts skills from resumes and the job description using Named Entity Recognition.
RAG-based Skill Proficiency Scoring: Uses a Retrieval-Augmented Generation approach to prompt a language model for proficiency scoring based on the skills detected.
Job Description Comparison: Compares skills in the resumes to those listed in the job description and provides a score based on relevance.
Output: Scores for each skill in the resumes, highlighting the proficiency and relevance to the job description.
