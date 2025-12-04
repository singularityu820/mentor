An LLM-powered multi-agent framework for goal-oriented learning in intelligent tutoring systems.

Tianfu Wang, Yi Zhan, Jianxun Lian, Zhengyu Hu, Nicholas Jing Yuan, Qi Zhang, Xing Xie, Hui Xiong

Tianfu Wang HKUST (GZ) Guangzhou, China tianfuwang.cs@outlook.com

Yi Zhan Microsoft Inc. China yzhan0119@outlook.com

Jianxun Lian Microsoft Research Asia Beijing, China jianxun.lian@outlook.com 

Zhengyu Hu HKUST (GZ) Guangzhou, China zhu021@connect.hkust-gz.edu.cn

Nicholas Jing Yuan* Microsoft Inc. China nicholas.yuan@microsoft.com

Qi Zhang Microsoft Inc. China zhang.qi@microsoft.com 

Xing Xie Microsoft Research Asia Beijing, China xingx@microsoft.com 

Hui Xiong* HKUST (GZ) Guangzhou, China xionghui@ust.hk

## ABSTRACT

Intelligent Tutoring Systems (ITSs) have revolutionized education by offering personalized learning experiences. However, as goal-oriented learning, which emphasizes efficiently achieving specific objectives, becomes increasingly important in professional contexts, existing ITSs often struggle to deliver this type of targeted learning experience. In this paper, we propose GenMentor, an LLM-powered multi-agent framework designed to deliver goal-oriented, personalized learning within ITSs. GenMentor begins by accurately mapping learners' goals to required skills using a fine-tuned LLM trained on a custom goal-to-skill dataset. After identifying the skill gap, it schedules an efficient learning path using an evolving optimization approach, driven by a comprehensive and dynamic profile of learners' multifaceted status. Additionally, GenMentor tailors learning content with an exploration-drafting-integration mechanism to align with individual learner needs. Extensive automated and human evaluations demonstrate GenMentor's effectiveness in learning guidance and content quality. Furthermore, we have deployed it in practice and also implemented it as an application. A practical human study with professional learners further highlights its effectiveness in goal alignment and resource targeting, leading to enhanced personalization. Supplementary resources are available at https://github.com/GeminiLight/gen-mentor.

### CCS CONCEPTS

* Information systems → Web mining.

### KEYWORDS

Intelligent Tutoring System, Large Language Model, Multi-agent

## 1 INTRODUCTION

Intelligent Tutoring Systems (ITSs) have made significant strides in supporting personalized learning experiences by leveraging machine learning (ML) technologies. These systems increasingly utilize data-driven insights across key aspects, such as managing learning materials, profiling learner status, and delivering personalized feedback. However, many ML-based ITSs suffer from fragmentation due to technical inconsistencies across modules. They also exhibit limited adaptability to emerging topics, often requiring extensive retraining to accommodate new educational topics. Recently, large language models (LLMs) have presented new opportunities to enhance ITS by offering interactive feedback through conversational interfaces. Existing studies on LLM-based ITS demonstrate their facilitation in content generation, query response, and problem-solving. However, while LLMs excel at dialogue-based engagement, they often remain reactive, merely responding to learner queries without proactively guiding learners toward their objectives. This reactivity also limits their capacity to gain a comprehensive understanding of learners, thus weakening the personalization.

As educational demands continue to diversify, especially within professional and lifelong learning contexts, learners increasingly seek systems that support personal or career-specific goals. For example, an employee assigned a task or seeking new jobs that involve unfamiliar technology may feel uncertain about where to start or what specific skills to develop. Without clear guidance, they risk becoming overwhelmed or wasting time on less relevant content. Such learners would benefit from a system that bridges this gap by quickly identifying their knowledge deficiencies and providing targeted content to help them acquire the precise skills needed to complete the objective efficiently. This personalized approach, known as goal-oriented learning, goes beyond merely delivering information, focusing on guiding goal achievement. However, traditional ITSs often rely on static curricula, offering broad content that may not adequately address the unique needs of individual learners. Additionally, LLM-based dialogue ITSs tend to be reactive rather than actively guide learners to achieve goals. As a result, these systems struggle to provide goal-oriented, personalized guidance, especially in fast-paced, professional contexts.

To address these gaps, we propose a perspective shift toward goal-oriented learning for transforming ITSs. As illustrated in Figure 1, unlike traditional ITSs that rely primarily on static curricula or reactive engagement, our approach emphasizes proactively guiding learners to achieve their specific goals. In goal-oriented ITSs, the learning process is driven by the learner's objectives rather than a predefined curriculum. By identifying skill gaps and pro-filing learner status, these systems can deliver more personalized pathways and tailored content, enabling learners to rapidly acquire the necessary skills to meet their specific goals. Notably, LLMs hold promising potential due to their remarkable ability to understand complex intentions and generate versatile content, making them well-suited for goal-oriented learning experiences.

Despite these benefits, several challenges should be addressed to ensure the effective functioning of LLM-powered goal-oriented ITSs. First, accurately identifying the learner's skill gaps in relation to their specific goals is critical, i.e., mapping the learner's objectives to the newly required skills for progress. This necessitates that LLMs handle a nuanced understanding of both the learner's profile and the target goal. Second, robust learner modeling is crucial for capturing individual preferences, progress, and cognitive abilities. The system must continuously adapt to real-time feedback, ensuring the learning experience evolves with the learner's needs and abilities, while dynamically adjusting strategies based on the learner's state. Third, providing personalized resources, e.g., both learning paths and delivered content, facilitates efficient knowledge acquisition. The system must ensure that resources are relevant, goal-aligned, and adaptable to learner progress, presenting challenges in content curation and learning path optimization.

In this work, we introduce GenMentor, an LLM-powered multi-agent ITS designed to deliver a personalized, goal-oriented learning experience. To address the diverse functions required in ITSs, we distribute responsibilities across multiple LLM agents, allowing them to collaboratively manage distinct tasks. Specifically, to improve the relevance and goal alignment of identified skills, we customize a goal-to-skill dataset to fine-tune the LLM as skill identifier, which contributes to accurately mapping learning goals to the necessary skills. Then, this agent identifies the skill gap by additionally considering learner's initial information. To enable learner-specific customization, we employ an adaptive learner profiler to dynamically track the learner's cognitive status, preferences, and behavioral patterns. By integrating real-time interactions, it also continuously updates the learner's profile for a deeper and more accurate understanding. Furthermore, to close the identified skill gap, the path scheduler continuously refines the learning path for effective goal achievement, where a learner simulator with generated profile is used to mimic learner feedback on provided resources. We also implement an content creator with an exploration-generation-integration mechanism, i.e., first exploring goal-oriented knowledge and preparing an outline, then drafting each section, and finally integrating and refining them to create learning materials. This process ensures that delivered content remains aligned with the learner's goals while enhancing personalization and effectiveness.

To evaluate the effectiveness of GenMentor, we conduct both automated and human evaluations, which show its superiority in identified skill requirements, scheduled learning paths, and generated content. Additionally, we have deployed it in the practical product and implemented an independent application designed for professional employee learning and goal achievement. A human study further highlights its effectiveness in providing learning guidance and targeting resources, and facilitating goal achievement. We summarize the main contributions of our work as follows:

* We introduce a novel perspective on ITS, goal-oriented learning, to meet the practical educational needs of learners. Through the integration of LLMs, we propose GenMentor, an LLM multi-agent system to implement this philosophy.
* To improve goal alignment of identified skills, we build a goal-to-skill mapping dataset for LLM fine-tuning. Grounded in educational theories, we propose a adaptive learner modeling method to capture evolving needs. We design an evolvable learning path scheduling and an exploration-draft-integration method to enhance resource personalization.
* We conduct extensive experiments through both automated and human evaluation. The results demonstrates that GenMentor offers superior learning guidance and resources.
* We deploy GenMentor in a practical product and implement it as an application. A study with professional learners also shows its effectiveness in achieving goal-oriented learning.

## 2 RELATED WORKS

### 2.1 Intelligent Tutoring System

Intelligent Tutoring Systems (ITSs) are designed to provide personalized experiences, mimicking one-on-one human tutoring. With advances in machine learning, modern ITSs have shifted toward leveraging data-driven approaches for material management, learner modeling, and feedback provision. Despite these advancements, existing ML-based ITSs often combine several ML-based models for distinct tasks, resulting in technical inconsistencies and data fragmentation, and lacking generalizability for emerging topics. Additionally, they rely on static content and predefined curricula, focusing on general knowledge acquisition rather than dynamically adjusting to individual learner goals. Recently, several works have integrated LLMs into ITS as a conversational tutor due to their excellent abilities in understanding and generation of LLMs. Otherwise, these systems can not proactively guide learners toward their long-term objectives due to their reactive nature and dialogue-based engagement. In our work, we explore a unified LLM-based multi-agent framework for ITS, which addresses the fragmentation that often arises from separate traditional ML models for ITS and enhances consistency across tutoring phases. Furthermore, we focus on providing proactive learning guidance for goal achievement, which differs from traditional ITS for broad knowledge acquisition.

### 2.2 LLM-powered Multi-agent Systems

Unlike individual LLM agents that operate independently, LLM-based multi-agent systems employ multiple LLM agents with diverse profiles to collaboratively tackle tasks. By distributing responsibilities, these systems can address more dynamic and complex challenges. This approach has shown promising results across various domains, such as software development, embodied agent and game playing. For instance, in software development, a chat-powered framework was proposed where LLM agents collaborate across different phases, (i.e., design, coding, and testing), through unified language-based communication. Additionally, LLM multi-agents have been explored in society simulation, where they simulate human behaviors and provide feedback, mimicking real-world interactions. These studies highlight the human-like capabilities of LLM agents and offer valuable insights into human behavior and decision-making. In the context of ITSs, they involve complex and evolving tasks such as personalized content generation, adaptive learning path design, and real-time student feedback processing. To handle such intricacies and enhance overall consistency, we leverage the collaborative power of LLM agents in ITS stems from their ability to handle multiple specialized roles, enabling more personalized and adaptive learning experiences.

## 3 PROBLEM STATEMENT

In goal-oriented learning, the focus is on achieving specific objectives efficiently, such as completing a project or mastering job-related skills. This requires proactive guidance to avoid unfocused learning and to improve goal completion efficiency. A goal-oriented ITS aims to customize learning pathways and content, enabling learners to quickly acquire the knowledge needed to meet their specific goals. Formally, let U₀ = (S₀, P, B) represent the learner's profile, where S₀, P, B denote the learner's initial knowledge status, preferences, and behavior patterns, respectively. Given a specific goal G, the learner must master a set of skills S' necessary to achieve this objective. Our aim is to efficiently minimize the skill gap ΔS = S' - S₀ with a personalized and adaptive learning experience by scheduling a personalized learning path L and tailored contents C based on P, B, and real-time learner interactions I. To accomplish this, the system should address three key sub-tasks:

* **Skill Gap Identification.** This step identifies the skills gap between the learners' current knowledge and the skills required to achieve their goals. First, goal G is mapped to the necessary skills, f : G → S', and then the skill gap ΔS₀ is identified by f : (S₀, S') → ΔS₀, where the S₀ is derived from learner's provided individual information I₀.
* **Adaptive Learner Modeling.** This module continuously updates learner profiles by incorporating interaction data It at timestep t, yielding f : (Ut-1, ΔS₀, It) → (Ut, ΔSt). This enables the system to track cognitive progress, recognize learning preferences, identify and behavior patterns.
* **Personalized Resource Delivery.** To efficiently close the skill gap ΔSt, this module dynamically schedules an engaging learning path Lt and delivers tailored learning content Ct, i.e., f : (Ut, ΔSt) → (Lt, Ct). The path should adapt to the learner's evolvable progress and preferences, while the content should be high-quality, goal-relevant and personalized.

## 4 THE GENMENTOR FRAMEWORK

In this work, we propose an LLM-powered multi-agent framework for goal-oriented learning in ITS, named GenMentor. To address the above intricate sub-tasks, we distribute responsibilities across multiple LLM agents, allowing them to collaboratively manage different tasks. As illustrated in Figure 2, this system begins with accurately assessing the skill gap between the learner's current skills and the target objective. Once this gap is identified, the ITS generates a personalized learning path with an evolvable optimization method for both learning efficiency and engagement. Furthermore, the system curates and generates content that is goal-oriented, up-to-date, and tailored to the learner's specific needs, ensuring the learner focuses on the most relevant and targeted content. During the learning process, the generated learner profile are continuously adjusted with the newly learner's interactions, enabling dynamical adaptation to the learner's evolving progress, preferences and needs.

### 4.1 Skill Gap Identification

To personalize the learning experience toward achieving specific goals, we identify the skill gap, i.e., the necessary skills bridging the learner's current cognitive status and target objectives. Mapping these goals accurately requires the LLM to grasp the goal's nuances, given their often abstract and high-level nature. However, direct prompting LLMs may produce irrelevant, unnecessary or incomplete skills, impeding effective goal achievement. Thus, we build a customized goal-to-skills dataset with Chain of Thought (CoT) to fine-tune LLMs, improving the goal alignment of identified gap.

#### 4.1.1 CoT-enabled Dataset Construction.

Given the absence of a directly relevant dataset for our task, we turn to job posting datasets, which contain detailed information on job roles, descriptions, and required skills. These datasets provide foundational insights into the expectations and requirements for various roles. By extracting pairs of job summaries (comprising roles and brief job descriptions) and their corresponding core skills, we construct a goal-to-skill dataset, treating these job summaries as goals. However, direct fine-tuning on these datasets may fall short in accurately mapping goals to skills, largely due to the abstract and high-level language common in job descriptions. To address this limitation, we employ CoT reasoning, introducing intermediate steps that clarify the logical connections between job responsibilities and the necessary skills. The CoT process involves breaking down the goals into key tasks, identifying the required skills for each task, and determining the proficiency levels needed, producing samples of <job summary, reasoning tracks, required skills>. See Appendix A.1 for more details. This approach facilitates capturing the nuanced relationships between goals and skills, thereby improving fine-tuning accuracy.

#### 4.1.2 Fine-tuning LLM for Goal Alignment.

Using the constructed goal-to-skills dataset, we fine-tune the LLM as skill identifier to accurately map goals to specific skills. This step ensures that this agent identifies relevant and complete skills while filtering out unnecessary skills, focusing on efficient goal achievement.

#### 4.1.3 Gap Identification Process.

This process begins with goal-to-skills mapping, S' = LLMskill-identifier(G), where the learning goal is mapped to a set of skills that are the core competencies needed for the goal. Next, identified skills are compared with learner's initial cognitive status based on provided information to establish the skill gap, ΔS = LLMskill-identifier(S', S₀). This step filters out already-mastered skills and highlights areas needing improvement.

### 4.2 Adaptive Learner Modeling

Understanding the learner's status is essential for ITS as it personalizes learning by adapting content to their needs, and providing targeted feedback. Instead of traditional ML-based methods lacking generalization and integration while LLM-based dialogue remain reactive, here, we explore how to explicitly leverage LLM to achieve the comprehensive and dynamic learner profile U.

#### 4.2.1 Comprehensive Learner Profile.

To capture learner's knowledge status and provide customized learning resources that align with goals and preferences, we consider three fundamental aspects to create a comprehensive learning profile U, informed by educational theories. These aspects are as follows:

* **Cognitive Status S.** To monitor the learner's knowledge acquisition, we track learning progress and assess mastery of required skills. We represent these skills as a set of competencies the learner has mastered and those they still need to acquire, along with corresponding metrics of progress and mastery. This approach highlights remaining skill gaps and enables the system to provide targeted content to help the learner achieve their goals.
* **Learning Preferences P.** Recognizing the diverse ways learners absorb information, this aspect captures individual preferences such as preferred content styles (e.g., concise summaries, detailed explanations) and preferred activity types (e.g., reading, active querying, interactive exercises). These insights enable the system to adapt its instructional methods and dynamically adjust the content delivery to enhance learner engagement and knowledge comprehension.
* **Behavioral Patterns B.** By analyzing interaction data, we aim to identify behavioral trends that affect learning engagement, such the system usage frequency and the time consumption variability of learning sessions. Infrequent use and irregular time consumption (e.g., long time spent in one session) may signal disengagement or difficulty. These insights allow for proactive interventions, such as motivational messages or adjusted content difficulty, to maintain learner momentum toward their goals.

#### 4.2.2 Dynamic Learner Modeling.

Initially, based on provided information of learner I₀ (e.g., resume) and identified skill gaps Δ, we use a learner profiler to create a preliminary profile, i.e., U₀ = LLMlearner-profiler(I₀, ΔS₀), capturing aforementioned three aspects. This initial profile serves as the starting point, may having limitations or inaccuracies. As the learner interacts with the system, their profile is continuously updated and refined based on collected dynamic interactions It (e.g., real-time performance data and proactive feedback) at each timestep t, i.e., U₁ = LLMlearner-profiler(Ut-1, It). During each session learning at timestep t, the system collects learner feedback and tracks metrics like performance and time use as interaction data It. For learning preferences P, data on time spent on different activities and feedback identify favored content, enabling dynamic adjustment of instructional content. For example, if a learner favors interactive exercises, the system prioritizes these activities by offering more exercises. Regarding behavioral patterns B, platform usage frequency and engagement consistency help gauge motivation. For example, irregular patterns (e.g., prolonged durations on single sessions or infrequent login) prompt interventions like trigger motivational prompts or adjusted content difficulty to maintain engagement and prevent frustration. After each session, the system assesses the progress of cognitive status by quiz scores and learner-reported feedback. It updates the learning process and skill mastery and identifies the remaining skill gap for future learning. An illustrative example is shown in Figure 3.

### 4.3 Personalized Resource Delivery

To effectively close identified skill gaps, GenMentor employs a personalized, adaptive content delivery method that dynamically aligns resources with each learner's unique profile and progress. Concretely, given the skill gap ΔS₀ and learner profile U₀ obtained in the previous stage, the path schedule creates a learning path L = {l₁, l₂, ..., ln}, consisting of multiple learning sessions li. For each learning session li, content creator tailor content Cᵢ with an exploration-drafting-integration mechanism to improve the resource personalization and targeting. During the learning process, a learner simulator using the dynamic learner profile Uₜ is employed to mimic the learner feedback on learning resources for refinement, enabling the adaptation of the learning path and content.

#### 4.3.1 Learner Simulator via Adaptive Profiling.

To achieve adaptations of learning resources while minimizing reliance on direct user feedback, we design an LLM agent-based feedback mechanism that simulates the learner responses. Specifically, leveraging real-time learner profiles Ut, a learner simulator employs the role-playing method to anticipate learner feedback for delivered resources. This simulation serves as a proxy to optimize delivered resources without requiring direct learner feedback. For learning paths, the learner simulator evaluates factors such as efficiency, engagement, and task difficulty, while anticipating learner reactions to tailored content (e.g., asking for more exercise, or feedbacking too high difficulties). This method allows the system to proactively adjust the learning resource to better match learner intentions and preferences, maximizing comprehension and sustaining motivation.

#### 4.3.2 Evolvable Learning Path Scheduling.

We use a path scheduler equipped with CoT reasoning ability to plan effective and engaged learning pathways that effectively support skill acquisition. Initially, it constructs an initial learning path L₀ = LLMpath-scheduler(U₀, ΔS₀) based on the identified skill gap ΔS₀ and the initialized learner profile U₀. This path is refined iteratively with feedback from the learner simulator, ensuring the learning path becomes progressively challenging and motivational for the learner. As the learning process progresses, the learner profile Uₜ is continuously updated to incorporate insights into the learner's evolving preferences, abilities, and progress. After completing a session at timestep t, the path scheduler dynamically re-evaluates and adjusts the learning path Lt-₁ with the updated profile, while generating Lt = LLMpath-scheduler(Ut, ΔSt, Lt-₁). If the learner approves, the updated learning path Lt replaces the previous one. By iteratively evaluating and adjusting the learning path, this adaptive scheduling mechanism ensures alignment with learner's evolving needs.

#### 4.3.3 Tailored Content Curation.

To offer learners relevant, personalized learning content, GenMentor employs a content creator with an exploration-drafting-integration mechanism to curate materials, including documents and quizzes. To ensure accuracy and up-to-date information, the content creator integrates web search tools for retrieval-augmented generation (RAG). Specifically, for a given learning session li, the content creator begins with the exploration to identify goal-related knowledge points across diverse perspectives. It then drafts content based on a systematic document outline considering the learner's preferences, and finally integrates feedback from the learner simulator to refine and finalize the learning material. This approach ensures that the content remains accurate and learner-focused, facilitating goal achievement.
Goal-oriented Knowledge Exploration. Key knowledge points are identified to ensure the learning content is both comprehensive and aligned with the learner's goals. Guided by practical learning, the content covers foundational concepts that provide the necessary background knowledge, practical insights that help bridge the gap between theoretical knowledge and real-world application, and problem-solving strategies that equip learners with techniques to address challenges, fostering critical thinking and adaptability. The content creator uses the session title as a query, to retrieve the latest information. Combining this up-to-date information with the inherent knowledge of the LLM, it identifies the relevant knowledge points and organizes them into a structured document outline. This outline, derived by oᵢ = LLMcontent-creator(Ut, ΔSt, Lt, li), serves as the foundation for subsequent drafting, ensuring that the content remains targeted, current, and aligned with the learner's objectives.
RAG-based Section Drafting. When drafting content of each section, to mitigate common issues in direct LLM generation such as hallucinations and long-tail inaccuracies, the content creator integrates high-quality retrieved information. It formulates queries by combining the session title and section titles to retrieve relevant and reliable data. This retrieved information is then re-customized to align with the learner's profile, ensuring the content is both informative and engaging. These section drafts, dᵢ = LLMcontent-creator(Ut, ΔSt, Lt, li, oᵢ), are tailored to match the learner's preference and progress, improving the personalization.
Integration and Refinement. The section drafts are synthesized into a cohesive document and then the learner simulator provides the mimicked feedback of learner and assesses logical structure and coherence. The content creator refines sections requiring improvement, ensuring each part aligns with the learner's needs. Once refined, these sections are seamlessly integrated into the final learning document. Quizzes are generated alongside the document to enhance learner engagement and test knowledge mastery. This process is denoted as cᵢ = LLMcontent-creator(Ut, ΔSt, Lt, li, oᵢ, dᵢ). This method further ensures the final content is logically organized and learner-tailored, promoting both comprehension and motivation.

## 5 EXPERIMENTS

To evaluate the effectiveness of GenMentor's output items, we conduct both LLM-based automated and human evaluations.

### 5.1 Implementations and Settings

We implement GenMentor with two popular LLMs, GPT-4o (2024-08-06) and Llama 3.2 (3B). For the skill identifier, both two LLMs are fine-tuned in Azure AI Studio on our custom goal-to-skill dataset, using a batch size of 3, and a maximum of 10 epochs. content creator uses the Bing search tool to access the internet, retrieving up to 5 results per query to ensure concise and relevant information. In the RAG module, the text embeddings are generated by the text-embedding-3-small model. We set the LLM temperature of 0.7 across all experiments to balance consistency and creativity.

### 5.2 LLM-based Automated Evaluation

Following prior works, we use GPT4o as an automated evaluator due to its strong alignment with human judgments. We adopt the 5-point Likert scale assessment to evaluate outputs.

#### 5.2.1 Overall Experiment Setup.

We adopt a resume dataset¹ to represent diverse learner information and a subset of a job posting dataset to define target learning goals (excluded from the skill identifier model's training data). To construct the testing dataset, we randomly matched resumes with job postings to simulate the input information provided by learners and predefined goals. The testing dataset included five types of occupations, with a total of 200 samples. For each sample, we simulate a learner with a specific resume as learner information S₀, aimed to acquire the skills required for a target job position G. We focus on three key output items: identified skills, learning path and learning content, measured by different metrics that impact on learning experience.

¹https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

#### 5.2.2 Evaluating Goal-to-skill Mapping.

We compare GenMentor with three approaches: (1) Direct Prompt (DirPrompt) and (2) CoTPrompt, where the CoT reasoning is integrated into the prompt no or yes. Lastly, we include a variant of GenMentor, (3) w/o Tracks, which removes track-based guidance to assess its impact. All methods use the same learner information and learning goal as input. To evaluate the quality of goal-to-skill mapping, we regard the skill requirements extracted from job postings as ground truth and use LLM to measure three metrics: Recall, the proportion of identified skills out of ground truth, assessing the comprehensiveness; Precision, the necessity by calculating the proportion of correct skills among the generated outputs; and Goal Alignment, indicating how well the identified skills align with the given learning goal.

As shown in Table 1, GenMentor with GPT-40 outperforms all baselines across all metrics, achieving a Recall of 0.67, Precision of 0.63, and Goal Alignment of 4.28, demonstrating its ability to generate comprehensive, accurate, and goal-aligned skill mappings. The track-based guidance in GenMentor is useful in most metrics, as removing it reduces Goal Alignment to 4.05, highlighting the importance of structured guidance. While CoTPrompt, with its reasoning steps, performs better than DirPrompt, both baselines fall short of GenMentor, particularly in precision and alignment. These results underscore the effectiveness of GenMentor's CoT-enabled fine-tuning method for optimal goal-to-skill mapping.

#### 5.2.3 Evaluating Learning Path.

To assess the quality of the learning path scheduled by GenMentor, we evaluate its effectiveness in two dimensions: Progression, which measures the logical flow and scalability of difficulty, and Engagement, which assesses the degree to which the path keeps learners motivated and interested. Higher scores on these metrics indicate better performance. We compare GenMentor with DirPrompt and CoTPrompt, introduced above. They are given the same learner profile U₀ and skill gap S₀ initialized by GenMentor to schedule the learning path statically.

As shown in Table 2, GenMentor achieves the highest scores across both metrics, outperforming the baselines in both Progression (4.56) and Engagement (4.71) with GPT-40, and 4.09 and 4.32 with Llama, respectively. CoTPrompt performs moderately well but remains below GenMentor, while DirPrompt shows significantly lower performance. It demonstrates that the use of CoT prompts notable benefits for this planning task requiring effective reasoning. Comparing CoTPrompt without mimicked feedback reveals, we observe that the learner simulator plays a crucial role in further enhancing the quality of the scheduled learning path. These results underscore the importance of incorporating CoT reasoning and mimicked feedback to optimize learning path scheduling.

#### 5.2.4 Evaluating Learning Content.

We compare GenMentor with three baselines: (1) DirGen, which generates content directly without structured guidance; (2) RAG, which incorporates RAG via the search tool; and (3) OutlineRAG, which first prepares a document outline and then drafts each section for integration. Additionally, we consider a variation of GenMentor for comparison: (4) w/o Refinement that skips the refinement step, using initial section drafts as the final document. To assess the quality of learning content generated by GenMentor, we evaluate its performance across four key dimensions: Goal Relevance, Content Quality, Engagement, and Personalization. Given the same initialized learner profile U₀, skill gap ΔS₀ and learning path L₀ derived by GenMentor in the previous stage as input data, these methods produce a set of learning documents of each learning session.

As shown in Figure 4, GenMentor consistently outperforms the baselines across most metrics for both GPT-40 and LLaMA-based models, particularly excelling in Personalization. With GPT-40, GenMentor achieves notable scores in Personalization (4.17) and Content Quality (4.86), showcasing its ability to deliver high-quality content tailored to learners' preferences. While OutlineRAG and the w/o Refinement perform closely, they fall short in Personalization (3.79 and 3.85), indicating that the refinement step using mimicked learner feedback enhances customization. DirGen and RAG exhibit lower performance in multiple dimensions, which highlights the importance of a structured approach in delivering goal-relevant and high-quality content. With Llama-based models, GenMentor maintains its superior performance in Personalization (4.12) and Content Quality (4.62). OutlineRAG continues to deliver moderate results but is significantly outpaced by GenMentor in personalization (3.67). These findings demonstrate GenMentor's ability to align learning content with learner goals and profiles, offering a more personalized learning experience.

#### 5.2.5 Human Validation on Automated Evaluation.

To assess the quality of these results in automated evaluation, we compare automated scores with human grading on sampled results. The results show 5 out of 7 metrics exhibit a statistically significant positive correlation. Please see Appendix A.2 for details.

### 5.3 Human Preference Evaluation

To further evaluate the effectiveness of GenMentor, we conduct a pairwise human evaluation comparing it to strong baselines. For skill gap and learning path, the baseline method is CoTPrompt, while for learning content, the baseline used is OutlineRAG. The results are shown Table 5. We observe that GenMentor was more favored, showcasing its ability to produce high-quality outputs. See Appendix A.3 for detailed experimental setup and result analysis.

| | Recall | Precision | Goal Alignment |
| :--- | :--- | :--- | :--- |
| GPT4o | | | |
| DirPrompt | 0.42 | 0.31 | 3.45 |
| CoTPrompt | 0.48 | 0.39 | 3.51 |
| GenMentor | 0.67 | 0.63 | 4.28 |
| w/o Tracks | 0.63 | 0.67 | 4.05 |
| Llama | | | |
| DirPrompt | 0.37 | 0.35 | 3.18 |
| CoTPrompt | 0.45 | 0.38 | 3.24 |
| GenMentor | 0.63 | 0.61 | 4.14 |
| w/o Tracks | 0.61 | 0.58 | 4.01 |

Table 1: Evaluation results on goal-to-skill mapping.

| | Progression | Engagement |
| :--- | :--- | :--- |
| GPT4o | | |
| DirPrompt | 3.95 | 3.80 |
| CoTPrompt | 4.38 | 4.63 |
| GenMentor | 4.56 | 4.71 |
| Llama | | |
| DirPrompt | 3.94 | 3.71 |
| CoTPrompt | 4.07 | 4.17 |
| GenMentor | 4.09 | 4.32 |

Table 2: Evaluation results on Learning Path.

## 6 END-TO-END HUMAN STUDY

We have deployed GenMentor in practice and conducted the human study with professional learners for further evaluation.

### 6.1 Practical Deployment

We have deployed GenMentor within our private product, the AIEP platform, launched in October 2024. AIEP is designed to empower employees with AI-driven tools to enhance productivity and streamline skill development. In AIEP, GenMentor supports critical functionalities, including learning path scheduling, resource generation, and learner modeling management, all integrated through API-based interactions. Additionally, we have implemented GenMentor as an independent web-based application tailored for goal-oriented learning and personalized tutoring. This application enables learners to engage in various educational activities for goal achievement, such as skill gap identification, scheduling learning paths, and personalized content delivery. The user-friendly interface facilitates a highly interactive learning experience, allowing learners to achieve their goals efficiently and effectively. This application also has been adopted by employees at Microsoft and partner vendor companies, demonstrating its practical value in enhancing professional development. Please see Appendix A.4 for details on the application.

### 6.2 Human Study Procedure

To study GenMentor's practical effectiveness, we engaged 20 employees from diverse professional backgrounds, including 10 tech-related professionals (e.g., engineers, researchers) and 10 non-tech professionals (e.g., product managers, human resource specialists). Each participant had prior experience with our GenMentor application as well as traditional MOOC platforms and LLM-based chatbots (e.g., Microsoft Copilot, ChatGPT). The study included a questionnaire assessing output quality, learning efficiency, and user experience, followed by interviews to discuss GenMentor's strengths and limitations compared to existing tutoring systems.

### 6.3 Questionnaire Findings

Results are shown in the table 6 and our findings are as follows. Clear learning guidance for goal achievement (①②). 18 participants agreed that the system effectively identifies skills aligned with their goals, resulting in a high rating of 4.6 ± 0.8. The learning path offered by GenMentor was rated as 4.3 ± 0.8, with participants noting that it breaks down learning objectives into manageable steps. One participant highlighted, "The clear guidance made it easier to navigate addressing complex tasks and achieve goals." Personalized and contextually relevant content (③④). GenMentor's personalized content was well-received, with participants rating the generated content as useful and personalized, scoring 4.2 ± 1.0. However, the dynamic profile matching received a slightly lower rating of 4.1 ± 0.9, suggesting that there is room for improvement in refining the learning modeling approach. Participants commented on the generated learning materials, with one stating, “The content is well-structured and targeted, helping me stay focused. I find my name in the personalized content surprisingly." Improved goal-oriented learning experience (⑤-⑧). Participants reported significant improvements in learning efficiency, with more than 80% noting enhanced efficiency and high-level engagement in learning. Additionally, more than 14 participants agreed that GenMentor made them focused and facilitated goal achievement. One participant stated, "The system kept me engaged and motivated by adapting to my needs and providing timely feedback." Intuitive and User-friendly System Design (⑨-⑩). The user-friendly design of GenMentor is a supportive point in utility, with participants rating the system's ease of use at 4.6 ± 0.7. The overall satisfaction with the system also received a commendable score of 4.3 ± 0.7, with participants appreciating the seamless interaction and clear layout. A participant noted, "The interface is intuitive and easy to navigate, making the learning experience enjoyable."

### 6.4 Interview and Discussion

To gain deeper insights into GenMentor's potential, we conduct interviews comparing it with existing learning platforms and tutors.

#### 6.4.1 Comparison with Traditional MOOCs.

Compared to MOOC platforms, participants valued GenMentor for its ability to customize learning paths, narrow the learning focus, and enhance content personalization. 15 participants specifically highlighted its automated skill gap identification feature, which helps clarify learning needs and streamline content delivery. Additionally, the platform's dynamic content adjustment and progress tracking capabilities were praised for keeping learners engaged and facilitating efficient goal achievement (highlighted by 8 participants). Despite these strengths, participants suggested areas for improvement, such as incorporating more diverse and interactive content formats (noted by 13 participants) and enhancing the depth of subject matter (noted by 12 participants). Furthermore, Participants identified scenarios where GenMentor excels, such as providing personalized guidance for specific skill development (highlighted by 15 participants) (noted by 13 participants). While MOOCs offer broad-topic coverage through systematic general curricula, GenMentor's tailored and adaptive approach clearly outperforms MOOCs in scenarios requiring focused, goal-oriented learning.

#### 6.4.2 Comparison with Search-enhanced Chatbots.

Participants identified GenMentor's key advantages over search-enhanced chatbots in providing personalized learning paths and interactive guidance for task-specific learning (highlighted by 17 participants). 13 participants praised its clear and structured learning progression, which uniquely aligns content with learners' goals. Unlike the generic and reactive responses of chatbots, GenMentor reduces proactively asking of learners and fostering focused learning, while dynamically adjusting its content based on real-time learner progress (noted by 9 participants). While GenMentor outperformed chatbots in goal-oriented learning experience, participants suggested improvements, such as adding more diverse multimedia resources (noted by 8 participants) and optimizing system responsiveness to enhance real-time interactions (noted by 7 participants). Participants found GenMentor particularly advantageous for scenarios like breaking down complex topics into manageable learning paths (noted by 16 participants). Although chatbots may excel in quick problem-solving, GenMentor's structured, goal-driven approach delivers a deeper, more interactive learning experience.

## 7 CONCLUSION

In this paper, we presented GenMentor, an LLM-powered multi-agent framework for goal-oriented learning in ITS, designed to deliver highly personalized, goal-oriented learning experiences. GenMentor advances beyond existing ITSs by proactively guiding learners efficiently toward their goals through accurate skill gap identification, adaptive learner profiling, and personalized resource delivery. Through extensive evaluations, including automated and human evaluation, GenMentor demonstrated superior performance in key outputs, including identifying skill gaps, aligning learning paths, and delivering tailored content. Real-world deployment and user studies further validated its effectiveness in fostering efficient and personalized learning in professional contexts. Overall, with its adaptive design and learner-centric approach, GenMentor showcases the transformative potential of LLMs in advancing personalized and goal-oriented education.

## REFERENCES

 Vibe Aarkrog and Bjarne Wahlgren. Goal orientation and decision-making in education. Vocations and Learning, 15(1):71–86, 2022.
 Abir Abyaa, Mohammed Khalidi Idrissi, and Samir Bennani. Learner modelling: systematic review of the literature from the last 5 years. Educational Technology Research and Development, 67:1105–1143, 2019.
 Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
 Ali Alkhatlan and Jugal Kalita. Intelligent tutoring systems: A comprehensive historical survey with recent developments. arXiv preprint arXiv:1812.09628, 2018.
 Gaurav Arora, Shreya Jain, and Srujana Merugu. Intent detection in the age of llms. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, EMNLP, 2024.
 Darren Cambridge. Eportfolios for lifelong learning and assessment. John Wiley & Sons, 2010.
 Yulin Chen, Ning Ding, Hai-Tao Zheng, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Empowering private tutoring by chaining large language models. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, CIKM, 2024.
 Menna Fateen and Tsunenori Mine. Developing a tutoring dialog dataset to optimize llms for educational use. arXiv preprint arXiv:2410.19231, 2024.
 Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multiagents: A survey of progress and challenges. In Proceedings of the 33rd International Joint Conference on Artificial Intelligence, IJCAI, 2024.
 Huan He, Qinghua Zheng, and Bo Dong. Learnerexp: Exploring and explaining the time management of online learning activity. In The World Wide Web Conference, WWW, page 3521–3525, 2019.
 Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to learn long-tail knowledge. In International Conference on Machine Learning, ICML, pages 15696–15707, 2023.
 Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. Prometheus 2: An open source language model specialized in evaluating other language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP, 2024.
 Huiyong Li, Rwitajit Majumdar, Mei-Rong Alice Chen, and Hiroaki Ogata. Goal-oriented active learning (goal) system to promote reading engagement, self-directed learning behavior, and motivation in extensive reading. Computers & Education, 171:104239, 2021.
 Yuan Li, Yixuan Zhang, and Lichao Sun. Metaagents: Simulating interactions of human behaviors for Ilm-based task-oriented coordination via collaborative generative agents. arXiv preprint arXiv:2310.06500, 2023.
 Qi Liu, Shiwei Tong, Chuanren Liu, Hongke Zhao, Enhong Chen, Haiping Ma, and Shijin Wang. Exploiting cognitive structure for adaptive learning. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD, page 627–635, 2019.
 Zhengyuan Liu, Stella Xin Yin, Geyu Lin, and Nancy F Chen. Personality-aware student simulation for conversational intelligent tutoring systems. arXiv preprint arXiv:2404.06762, 2024.
 Jakub Macina, Nico Daheim, Lingzhi Wang, Tanmay Sinha, Manu Kapur, Iryna Gurevych, and Mrinmaya Sachan. Opportunities and challenges in neural dialog tutoring. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, EACL, pages 2357–2372, 2023.
 Zhao Mandi, Shreeya Jain, and Shuran Song. Roco: Dialectic multi-robot collaboration with large language models. In 2024 IEEE International Conference on Robotics and Automation, ICRA, pages 286–299, 2024.
 Hyacinth S Nwana. Intelligent tutoring systems: an overview. Artificial Intelligence Review, 4(4):251–277, 1990.
 Benjamin D Nye, Dillon Mee, and Mark G Core. Generative large language models for dialog-based tutoring: An early consideration of opportunities and concerns. In LLM@ AIED, pages 78–88, 2023.
 Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, UIST, pages 1–22, 2023.
 Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim. Empowering personalized learning through a conversation-based tutoring system with student modeling. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, CHI, pages 1–10, 2024.
 Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL, pages 15174–15186, 2024.
 Murray Shanahan, Kyle McDonell, and Laria Reynolds. Role play with large language models. Nature, 623(7987):493–498, 2023.
 Yijia Shao, Yucheng Jiang, Theodore A Kanell, Peter Xu, Omar Khattab, and Monica S Lam. Assisting in writing wikipedia-like articles from scratch with large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL, 2024.
 Robert A Sottilare, Arthur Graesser, Xiangen Hu, and Heather Holden. Design recommendations for intelligent tutoring systems: Volume 1-learner modeling, volume 1. US Army Research Laboratory, 2013.
 Jianwen Sun, Fenghua Yu, Qian Wan, Qing Li, Sannyuya Liu, and Xiaoxuan Shen. Interpretable knowledge tracing with multiscale state representation. In Proceedings of the ACM Web Conference 2024, WWW, page 3265–3276, 2024.
 John Sweller, Paul Ayres, Slava Kalyuga, John Sweller, Paul Ayres, and Slava Kalyuga. Measuring cognitive load. Cognitive load theory, pages 71–85, 2011.
 Mouenis Anouar Tadlaoui, Souhaib Aammou, Mohamed Khaldi, and Rommel Novaes Carvalho. Learner modeling in adaptive educational systems: A comparative study. International Journal of Modern Education & Computer Science, 8(3), 2016.
 Shiwei Tong, Jiayu Liu, Yuting Hong, Zhenya Huang, Le Wu, Qi Liu, Wei Huang, Enhong Chen, and Dan Zhang. Incremental cognitive diagnosis for intelligent education. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD, pages 1760–1770, 2022.
 Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
 Jeroen JG Van Merriënboer and Paul A Kirschner. Ten steps to complex learning: A systematic approach to four-component instructional design. Routledge, 2017.
 Shen Wang, Tianlong Xu, Hang Li, Chaoli Zhang, Joleen Liang, Jiliang Tang, Philip S Yu, and Qingsong Wen. Large language models for education: A survey and outlook. arXiv preprint arXiv:2403.18105, 2024.
 Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
 Fangyuan Xu, Yixiao Song, Mohit Iyyer, and Eunsol Choi. A critical evaluation of evaluations for long-form question answering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL, 2023.
 Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the werewolf game. arXiv preprint arXiv:2310.18940, 2023.
 Yue Yun, Huan Dai, Rui An, Yupei Zhang, and Xuequn Shang. Doubly constrained offline reinforcement learning for learning path recommendation. Knowledge-Based Systems, page 111242, 2023.
 Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging Ilm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2023.
 Yan Zhuang, Qi Liu, Zhenya Huang, Zhi Li, Binbin Jin, Haoyang Bi, Enhong Chen, and Shijin Wang. A robust computerized adaptive testing approach in educational question retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR, pages 416–426, 2022.

## A APPENDIX

Due to page limitation, we provide supplementary resources at https://github.com/GeminiLight/gen-mentor in addition to the Appendix, including demo, prompts, data and more insights.

### A.1 Details on Goal-to-skill Dataset

To construct the goal-to-skill dataset, we use the LinkedIn job posting dataset from Kaggle², which contains more than 0.12 million job postings across various positions. We filter the dataset based on the word count, retaining only postings with at least 500 words. This process results in a refined subset of 58,064 postings. From this subset, we randomly sample 10,000 postings to create the training dataset and 200 postings as the validation dataset, ensuring balanced representation across position types. For each sample, we employ GPT4o to extract the job summaries paired with their corresponding skill lists. Regarding the extracted skill lists as output, we use the CoT-enabled completion method to incorporate the immediate reasoning steps. This process involves breaking down the goals into key duties, identifying the required skills for each duty, and determining the proficiency levels needed, resulting in samples of <job summary, reasoning tracks, required skills>. This curated dataset provides clear mappings from goals (job objectives) to skills (necessary competencies), used for fine-tuning gap identifier and evaluating outputs of goal-to-skill mapping. Examples of these samples are available in the supplementary resources.

²https://www.kaggle.com/datasets/arshkon/linkedin-job-postings

### A.2 Human Validation on LLM Scoring

To validate the quality of the LLM-based automated evaluation results, we conduct an experiment comparing automated scores with human grading for GenMentor's output. Concretely, we invite two experts with extensive experience in Python development to participate in this validation. For each output item (i.e., goal-to-skill mapping, learning path, and learning content), 20 data points, all related to the occupation type of Python developer, were randomly sampled from the automated evaluations. Each data point represented the automated evaluation score of one method's Two independent evaluators were tasked with providing human grading for the same data points, and their average scores were used as the human judgment benchmark. We calculate the Pearson correlation between the automated evaluation scores and the human grading scores to assess the alignment between the two scores. As depicted in Table 3, the results show that 5 out of 7 evaluation metrics exhibit a statistically significant positive correlation between automated scores and human grading. This indicates the relative consistency of the LLM-based automated evaluation method and human sense.

### A.3 Human Preference Evaluation Details

#### A.3.1 Experiment Setup.

We invited 5 participants (including 3 software engineers and 2 product managers) to set learning goals related to occupations. Participants provided their resumes as learner information, which, along with each learning goal, served as input for both GenMentor and the baseline method. Each participant repeated this process 6 times for the distinct learning goals. The evaluation followed a step-by-step process. Both systems simultaneously generated skill gaps based on the provided information and learning goal. Using the preferred skill gap as additional input, two methods were used to generate corresponding learning paths. Finally, learning content was produced by both methods based on the same learning path. At each stage, participants reviewed the outputs side-by-side and selected their preferred option.

#### A.3.2 Results and Analysis.

As illustrated in Figure 5, results reveal that GenMentor outperformed the baseline systems across all evaluation items, further showing the effectiveness of the automated scores. For skill gap identification, GenMentor was preferred in 22 out of 30 cases, demonstrating its superior ability to align skills with learning goals compared to CoTPrompt. For learning path scheduling, GenMentor was chosen in 17 cases, reflecting its effectiveness in generating goal-oriented paths. For learning content generation, GenMentor was favored in 80% of cases, showcasing its ability to produce high-quality and personalized materials.

### A.4 Web-based Application of GenMentor

To illustrate the practical utility of GenMentor, we provide key interfaces of its web application in Figure 7. Learners begin on the onboarding page by setting their learning goals and providing their background information. GenMentor then identifies skill gaps, which learners can review and confirm. Based on the confirmed gaps, a personalized learning path is scheduled, with options for learners to manually adjust or allow automated refinements based on their updated profiles. During each learning session, tailored content is delivered to learners, ensuring alignment with their goals. Learners can also review and update their profiles on the learner profile page, maintaining control over their learning journey. This streamlined design underscores GenMentor's ability to deliver adaptive and goal-oriented learning experiences.

| Category | Metric | Correlation | p-value |
| :--- | :--- | :--- | :--- |
| Goal2Skill Mapping | Goal Alignment | 0.51 | < 4⁻² |
| Learning Path | Progression | 0.47 | < 2⁻² |
| | Engagement | 0.39 | < 3⁻¹ |
| Learning Content | Content Quality | 0.52 | < 1⁻² |
| | Goal Relevance | 0.46 | < 1⁻² |
| | Engagement | 0.38 | < 4⁻² |
| | Personalization | 0.42 | < 8⁻² |

Table 3: Pearson correlation between two types of scores.