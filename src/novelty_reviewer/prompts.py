"""Prompt templates.

The templates are copied verbatim from the runs that produced the dataset, the fine-tuned model and the results in
the paper; changing their wording changes model behaviour. Trailing spaces are intentional: do not strip them
(tests/test_prompt_fidelity.py checks every prompt).
"""

# ---------------------------------------------------------------------------------------------------------------------
# Novelty assessment of a manuscript (Novelty Reviewer)
# ---------------------------------------------------------------------------------------------------------------------

# Single user turn used to fine-tune the Novelty Reviewer model. Placeholder: {paper_text}.
NOVELTY_REVIEW_PROMPT = """
You are an expert reviewer for AI conferences. 
Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
You should carefully read the paper, judge whether the ideas are novel or not based on the technical content and the previous related work,
and provide a concise justification based only on the content of the paper.

Return two things: Novelty Score and Short Human-like Novelty Review.

Review the following paper for novelty:

{paper_text}

Provide your evaluation in the following format:
Novelty Score: 
-1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

Short Novelty Review: A paragraph of ~ 250 words containing your summary and reasoning about paper novelty/originality.
"""

# Prompt of the full framework: the manuscript plus the retrieved related work. Placeholders: {paper_text},
# {similar_papers}.
NOVELTY_REVIEW_WITH_CONTEXT_PROMPT = """
        You are an expert reviewer for AI conferences. 
        Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
        You should carefully read the paper, judge whether the ideas are novel or not, 
        and provide a concise justification based only on the content of the paper.

        Return two things: Novelty Score and Short Novelty Review.
        You MUST Provide your evaluation in the following format:
        Novelty Score: 
        -1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
        0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
        1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
        2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

        Short Novelty Review: A 3-5 sentence containing your summary and reasoning about paper novelty/originality.
        Review the following paper for novelty:

        {paper_text}

        take into account these similar papers:
        {similar_papers}

        
        """

# Prompt used for all baseline models in the paper. Placeholder: {paper_text}.
BASELINE_PROMPT = """
You are an expert reviewer for AI conferences. 
Your task is to evaluate the NOVELTY of research papers according to reviewer guidelines. Be factual, concise, and balanced.
You should carefully read the paper, judge whether the ideas are novel or not, 
and provide a concise justification based only on the content of the paper.

Return two things: Novelty Score and Short Novelty Review.
YOU MUST WRITE THE DIGIT FOR THE NOVELTY SCORE FIRST.
Review the following paper for novelty:

{paper_text}

Provide your evaluation in the following format:
Novelty Score: 
-1 = Not novel (work is incremental, derivative, or replicating existing approaches with minimal innovation.)
0 = Limited novelty (work is somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
1 = Moderately novel (some originality but overlap with prior work, or extension/combination of existing ideas.)
2 = Highly novel (fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

Short Novelty Review: A 3-5 sentence containing your summary and reasoning about paper novelty/originality.
Your response MUST follow the specified format exactly. The output should be around 250 words and end with a complete sentence with a followstop.

"""

# ---------------------------------------------------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------------------------------------------------

# Stage 1a: verbatim novelty excerpts of a single review (released as `novelty_excerpts`). Placeholder: {review_text}.
REVIEW_EXCERPTS_SYSTEM_PROMPT = """You are an expert scientific analyst specializing in peer review meta-analysis. Your task is to carefully read a peer review and extract all textual segments that bear on the novelty, originality, or contribution of the submitted paper, whether the reviewer expresses these judgments explicitly or implicitly. Do not paraphrase, summarize, or alter the extracted text in any way; return verbatim excerpts only."""

REVIEW_EXCERPTS_USER_PROMPT = """Given the peer review text below, extract all sentences or passages that assess the originality or contribution of the reviewed paper. Return your output in the exact JSON format specified. Apply the following extraction guidelines:

Extraction guidelines:
(1) Explicit novelty statements: Extract any sentence containing direct novelty-related vocabulary, including but not limited to: novel, original, new, innovative, first to, breakthrough, unprecedented, pioneering, creative, unique contribution.
(2) Implicit originality assessments: Extract sentences that evaluate the paper's relationship to prior work without using explicit novelty keywords. This includes statements such as:
    - Comparative positioning: "extends the work of X", "improves upon prior methods", "similar to [citation]"
    - Negative assessments: "incremental contribution", "lacks differentiation from existing approaches", "already explored in prior work"
    - Positive assessments: "significant advance over the state of the art", "fills an important gap"
(3) Contribution scope statements: Extract sentences that characterize the scope or significance of the paper's claimed contributions, including statements about whether the problem addressed is important, whether the proposed solution is meaningful, or whether the experimental gains are substantial.

Rules:
- Extract verbatim text only. Do not rephrase, merge, or truncate sentences.
- Include both positive and negative assessments — do not filter by sentiment.
- If a passage spans multiple sentences and cannot be understood in isolation, include the full passage.
- Do NOT extract descriptive sentences about what the paper does. Only extract evaluative judgments about originality.
- Do NOT extract reviewer questions. Only extract statements that clearly express an assessment of novelty or contribution.
- If no relevant segments are found, return an empty list [].
- Return only valid JSON with no preamble, explanation, or markdown formatting.

Output format (strict JSON, no additional text):
{{"novelty_excerpts": ["...", "..."]}}

[INPUT]
{review_text}
"""

# Stage 1b: per-review novelty stance and supporting quotes (input to the aggregation). Placeholder: {review}.
REVIEW_STANCE_SYSTEM_PROMPT = """You are an expert scientific assistant trained to assess academic peer reviews for discussions of novelty. Your task is to:

Analyze the reviewer's free-text feedback in the review field.

Identify explicit or implicit statements about the novelty of the paper.

Classify the novelty assessment as follows:

1 : The reviewer clearly or implicitly mentions the paper is novel.

-1 : The reviewer clearly or implicitly mentions the paper lacks novelty.

0 : There is no discussion about novelty.

When novelty is discussed (assessment = 1 or -1), include one or two brief supporting quotes from the reviewer that mention or imply novelty. If novelty is not discussed (assessment = 0), leave the quotes from the reviewer field as null.

Output must contain only the following 2 fields:

1. Novelty assessment: -1, 0, or 1.

2. Quotes from the reviewer: 1–2 short exact quotes from the review supporting the assessment, or null if not applicable.

"""

REVIEW_STANCE_USER_PROMPT = """

{review}

"""

# Stage 2: paper-level aggregation into the ECNA score and assessment. Placeholders: {reviews}, {novelty_excerpts}.
AGGREGATION_SYSTEM_PROMPT = """You are an expert reviewer for top-tier AI and ML conferences. 
Your task is to analyze paper reviews focusing on novelty aspects and synthesize a single, cohesive novelty assessment.
Given multiple peer reviews of a research paper, distill the collective perspective on novelty and originality into a unified evaluation written as direct statements about the work itself.
Be factual, concise, and balanced. 
Do not add your own judgment; only reflect the points expressed in the given reviews.
Write in a direct voice about the paper (e.g., "The paper introduces...", "The work extends...", "The approach combines...") rather than describing what reviewers said.
"""

AGGREGATION_USER_PROMPT = """
You will be provided with full review texts and pre-extracted novelty-related quotes and assessments from each review. 
Read and analyze carefully the following reviews for a research paper:

Full Reviews:
{reviews}

The following segments were identified as novelty discussions for each review:
{novelty_excerpts}

Assign a novelty score (-1–2) reflecting how reviewers collectively perceive the originality of the paper. 
Base the score only on their aggregated opinions.

-1 = Not novel (Reviewers consistently describe work as incremental, derivative, or replicating existing approaches with minimal innovation.)
0 = Limited novelty (Reviewers find the work somewhat standard, showing minor variations or applications of known methods without substantial conceptual or technical innovation.)
1 = Moderately novel (Reviewers acknowledge some originality but note significant overlap with prior work, or see it as a competent extension/combination of existing ideas.)
2 = Highly novel (Reviewers recognize fundamentally new ideas, approaches, problem formulations, or insights that significantly advance the field.)

Write your assessment as direct statements about the paper itself, NOT as a summary of what reviewers said.
AVOID phrases like: "Reviewers note that...", "They agree that...", "The reviewers find...", "According to reviewers..."
USE phrases like: "The paper introduces...", "The work presents...", "The approach builds on...", "The contribution lies in..."

If reviewers disagree significantly on specific aspects, present both perspectives as contrasting assessments of the work itself (e.g., "The method extends X, though the extensions remain largely incremental").

The output should be in the following format:

Novelty Score: [-1, 0, 1, 2]

Score Justification:
[2-3 sentences explaining the novelty level as direct statements about the paper's contributions and their originality]

Detailed Assessment:
[4-6 sentences written as direct statements about the paper covering:]
- The main novel contributions or new ideas introduced
- Limitations in originality or areas of incremental advancement
- Specific aspects of novelty: problem formulation, methodological innovations, experimental insights, or theoretical advances
"""

# ---------------------------------------------------------------------------------------------------------------------
# Retrieval-augmented originality analysis
# ---------------------------------------------------------------------------------------------------------------------

# Structured knowledge extraction from a manuscript or retrieved paper. Placeholders: {title}, {text}.
KNOWLEDGE_EXTRACTION_PROMPT = """Extract information from this research paper and return ONLY valid JSON.

            Title: {title}
            Content: {text}

            Return this JSON structure:
            {{
                "core_ideas": ["list of main ideas"],
                "methods": ["list of methods"],
                "contributions": ["list of contributions"],
                "keywords": ["list of keywords"],
                "data_sources": ["list of data sources"],
                "experiments": ["list of experiments"]
            }}

            Return ONLY the JSON, no explanation."""

# Short explanation of the overlap between the manuscript and a retrieved paper. Placeholders: {paper1}, {paper2}.
OVERLAP_EXPLANATION_PROMPT = """Compare these papers for ideas plagiarism and return ONLY valid JSON.

            {paper1}

            {paper2}

            Return this JSON:
            {{
                "overlap_percentage": 45,
                "similarity_aspects": "brief explanation"
            }}

            Return ONLY the JSON."""

# ---------------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------------

# LLM-as-a-judge score in [0, 10]. Placeholders: {novelty_summary}, {novelty_score}, {model_output}.
JUDGE_PROMPT = """Rate how well the Model Output aligns with the Reference Answer on a scale from 0 to 10,
        taking into account both the correctness and coverage of the novelty description and consistency with the provided Novelty Score.

        Reference Answer: {novelty_summary}
        Novelty Score: {novelty_score}

        Model Output: {model_output}

        Respond with ONLY a single number from 0 to 10."""
