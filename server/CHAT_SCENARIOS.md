# Ucotron Chat Scenarios: 20 Multi-Turn Scenarios for Cognitive Trust Testing

This document defines 20 comprehensive multi-turn chat scenarios across 7 industries to evaluate Ucotron's ability to maintain context, resolve contradictions, and recall information over extended conversations.

## Personal Assistant (4 scenarios)

### Scenario 1: Personal Profile Management
**Namespace**: `personal_profile`

| Turn | User Message |
|------|--------------|
| 1 | Hi! I'm Alex, 34 years old, currently living in San Francisco. I work in tech and love hiking on weekends. |
| 2 | My birthday is March 15th. I'm married to Jordan and we have two kids: Emma (8) and Lucas (5). |
| 3 | I should mention I have a mild dairy allergy and prefer vegetarian meals. |
| 4 | We're planning a family trip to Colorado next summer. Any suggestions for kid-friendly hiking trails? |
| 5 | Wait, who did I say my spouse's name was? (Recall test) |
| 6 | How old are my children again? (Recall test) |
| 7 | What did I mention about my dietary preferences? (Recall test) |

### Scenario 2: Daily Routine with Contradictions
**Namespace**: `daily_routine`

| Turn | User Message |
|------|--------------|
| 1 | I usually wake up at 6 AM and go to the gym before work. Work starts at 9 AM. |
| 2 | I have a standing meeting with my team every Tuesday at 10 AM. |
| 3 | Actually, let me correct that - I switched to working from home, so I wake up at 7 AM now. (Contradiction: wake time) |
| 4 | On Tuesdays, I need to attend that 10 AM meeting from my home office. But wait, I just realized I changed jobs last month and my new work starts at 8:30 AM. (Contradiction: start time) |
| 5 | What time did I say I wake up? (Recall test) |
| 6 | When does my work day begin? (Recall test) |
| 7 | Walk me through a typical Tuesday for me. (Recall test with contradiction resolution) |

### Scenario 3: Goals and Milestones
**Namespace**: `goals_milestones`

| Turn | User Message |
|------|--------------|
| 1 | I'm setting a goal to read 24 books this year - that's 2 per month. |
| 2 | I also want to run a half marathon by October and improve my Spanish to B1 level. |
| 3 | One more goal: I want to save $10,000 for a sabbatical in 2027. |
| 4 | I've already finished 3 books this year. Progress on Spanish is going well - I'm doing Duolingo daily. |
| 5 | How many books am I targeting this year? (Recall test) |
| 6 | What month was I planning to run that half marathon? (Recall test) |
| 7 | Tell me all my goals and my progress so far. (Comprehensive recall test) |

### Scenario 4: Travel Planning with Preferences
**Namespace**: `travel_planning`

| Turn | User Message |
|------|--------------|
| 1 | I'm planning a 2-week vacation to Spain in September. I love Mediterranean cuisine and historic architecture. |
| 2 | I prefer staying in boutique hotels rather than big chains. My budget is around $3,500 for the whole trip including flights. |
| 3 | I've been to Barcelona before, so I want to explore Madrid, Seville, and Granada this time. |
| 4 | Actually, I should mention - I'm a bit uncomfortable with very long flights. Under 12 hours ideally. |
| 5 | What destination am I visiting and how long will I be there? (Recall test) |
| 6 | What were my preferences regarding accommodation? (Recall test) |
| 7 | Summarize my travel plan with budget and preferences. (Comprehensive recall test) |

## Healthcare (3 scenarios)

### Scenario 5: Symptom Tracking and History
**Namespace**: `health_symptoms`

| Turn | User Message |
|------|--------------|
| 1 | I've been experiencing occasional headaches for the past month, usually in the afternoon. |
| 2 | The headaches are mild (3/10 severity) and often go away after I drink water. I think it might be dehydration. |
| 3 | Starting last week, I've also noticed I'm more tired than usual, especially after 3 PM. |
| 4 | I had a full blood panel done two weeks ago - everything came back normal. |
| 5 | What symptoms have I reported? (Recall test) |
| 6 | How severe were the headaches on a pain scale? (Recall test) |
| 7 | When did I have blood work done and what were the results? (Recall test) |

### Scenario 6: Medication Management with Contradictions
**Namespace**: `health_meds`

| Turn | User Message |
|------|--------------|
| 1 | I'm currently taking Lisinopril 10mg daily for blood pressure management. I've been on it for 2 years. |
| 2 | I also take Metformin 500mg twice daily for type 2 diabetes. No side effects so far. |
| 3 | Wait, I need to correct myself - I actually increased my Lisinopril to 15mg last month due to slightly elevated readings. (Contradiction: dosage) |
| 4 | Actually, my doctor said the Metformin dosage is now 750mg twice daily as of our last visit. (Contradiction: medication dosage) |
| 5 | What is my current Lisinopril dosage? (Recall test) |
| 6 | What medications am I taking and at what dosages? (Recall test with contradiction resolution) |

### Scenario 7: Medical Visits and Care Timeline
**Namespace**: `health_visits`

| Turn | User Message |
|------|--------------|
| 1 | I saw my primary care physician Dr. Smith on January 15th for my annual checkup. |
| 2 | During that visit, she recommended I get a flu shot and a tetanus booster. I got both done the following day. |
| 3 | I'm scheduled to see a cardiologist next month - February 28th - for a routine follow-up. |
| 4 | Dr. Smith also referred me to a nutritionist to help with managing my diet for diabetes prevention. |
| 5 | When and with whom did I have my last checkup? (Recall test) |
| 6 | What vaccinations did I get and when? (Recall test) |
| 7 | Summarize my healthcare timeline including appointments and recommendations. (Comprehensive recall test) |

## Legal (3 scenarios)

### Scenario 8: Case Documentation
**Namespace**: `legal_cases`

| Turn | User Message |
|------|--------------|
| 1 | I'm working on a property dispute case. It involves a boundary disagreement with my neighbor over 0.3 acres. |
| 2 | The case has been ongoing for 8 months. We hired attorney Maria González to represent us. |
| 3 | The disputed property is in Austin, Texas. My neighbor claims it belongs to them, but we have a 1987 deed that clearly shows it's ours. |
| 4 | Our next court date is March 20th, 2026. We're hoping for a favorable ruling based on the historical deed evidence. |
| 5 | Who is my attorney in this case? (Recall test) |
| 6 | Where is the disputed property located and what's the core disagreement? (Recall test) |
| 7 | Give me a complete summary of my legal case including timeline and next steps. (Comprehensive recall test) |

### Scenario 9: Legal Meetings and Strategy
**Namespace**: `legal_meetings`

| Turn | User Message |
|------|--------------|
| 1 | Met with my attorney yesterday to discuss the employment contract review. The company is offering me a Senior Developer role. |
| 2 | The base salary is $185,000 annually with a 15% performance bonus. Benefits include health insurance, 401k matching up to 6%, and 25 days PTO. |
| 3 | There's a non-compete clause that restricts me from working for competitors for 18 months after leaving. I'm concerned about this. |
| 4 | My attorney said the non-compete is reasonable for our industry but we can negotiate it down to 12 months. She's drafting a counter-offer. |
| 5 | What role is being offered to me? (Recall test) |
| 6 | What's my concern about the employment contract? (Recall test) |
| 7 | Summarize the job offer details and my attorney's recommendations. (Comprehensive recall test) |

### Scenario 10: Contract Review and Modifications
**Namespace**: `legal_contracts`

| Turn | User Message |
|------|--------------|
| 1 | I'm reviewing a freelance agreement with a web development client. The project scope includes website design, development, and 3 months of support. |
| 2 | Project fee is $25,000, with 50% due upfront and 50% on delivery. Timeline is 12 weeks. |
| 3 | I noticed the contract has undefined revision limits - they could ask for unlimited changes. I need to add a cap of 3 rounds of revisions. |
| 4 | Also, I want to add a clause about intellectual property - I'll retain ownership of tools and code libraries I create, but they get exclusive rights to the final website. |
| 5 | What's the total project fee? (Recall test) |
| 6 | What modifications did I want to make to the contract? (Recall test) |
| 7 | Present the contract terms with all my proposed modifications. (Comprehensive recall test) |

## Education (3 scenarios)

### Scenario 11: Student Progress Tracking
**Namespace**: `edu_students`

| Turn | User Message |
|------|--------------|
| 1 | I'm teaching a semester course with 28 students. It's an intermediate Python programming class meeting 3 times a week. |
| 2 | Sofia Martinez has been excellent - she's completed all assignments on time and her project proposals are innovative. Currently at 94% in the class. |
| 3 | I'm concerned about James Chen. He's missed 5 classes already, submitted only 2 of 8 assignments, and got a 62% on the midterm. |
| 4 | I'm planning to reach out to James next week to discuss his progress and available support options. |
| 5 | How many students are in my course? (Recall test) |
| 6 | How is Sofia performing compared to James? (Recall test) |
| 7 | Summarize the class roster issues and my action plan. (Comprehensive recall test) |

### Scenario 12: Curriculum Development (Bilingual)
**Namespace**: `edu_curriculum`

| Turn | User Message |
|------|--------------|
| 1 | Estoy diseñando un plan de estudios nuevo para Matemáticas Avanzadas, enfocado en estudiantes de 10-11 grado. |
| 2 | Los módulos principales son: Cálculo (5 semanas), Álgebra Lineal (4 semanas), Estadística (3 semanas). |
| 3 | I'm incorporating hands-on projects where students apply concepts to real-world problems like climate data analysis and financial modeling. |
| 4 | Evaluación será mediante proyectos (40%), exámenes parciales (35%), y participación en clase (25%). |
| 5 | What are the main modules in my curriculum? (Recall test) |
| 6 |¿Cuál es el enfoque pedagógico? (Recall test) |
| 7 | Give me a full curriculum overview with modules, duration, and assessment strategy. (Comprehensive recall test) |

### Scenario 13: Research Project Management
**Namespace**: `edu_research`

| Turn | User Message |
|------|--------------|
| 1 | I'm leading a research project on the effectiveness of personalized learning systems in secondary schools. We have 3 research partners and a $200,000 grant. |
| 2 | The project runs for 24 months. We'll have 15 schools participating with approximately 2,000 students total. |
| 3 | Our hypothesis is that adaptive learning systems improve student outcomes by 15-20% compared to traditional instruction. |
| 4 | We're collecting quantitative data (test scores, engagement metrics) and qualitative data (teacher interviews, student surveys). |
| 5 | How long is my research project? (Recall test) |
| 6 | How many schools and students are involved? (Recall test) |
| 7 | Describe my research project including hypothesis, funding, and data collection methods. (Comprehensive recall test) |

## Finance (3 scenarios)

### Scenario 14: Investment Portfolio Management
**Namespace**: `finance_portfolio`

| Turn | User Message |
|------|--------------|
| 1 | My investment portfolio consists of $400,000 allocated as follows: 60% stocks ($240,000), 30% bonds ($120,000), 10% cash ($40,000). |
| 2 | In stocks, I have $120,000 in US large-cap index funds, $80,000 in international funds, $40,000 in tech sector ETFs. |
| 3 | Bonds include $80,000 in corporate bonds (BBB rated), $40,000 in government bonds. Cash is split between emergency fund and money market. |
| 4 | My target allocation is to rebalance toward 50/40/10 (stocks/bonds/cash) over the next 6 months as I approach retirement in 5 years. |
| 5 | What's my total portfolio value? (Recall test) |
| 6 | How is my portfolio currently allocated? (Recall test) |
| 7 | Summarize my portfolio, current allocation, and rebalancing strategy. (Comprehensive recall test) |

### Scenario 15: Expense Tracking and Budgeting
**Namespace**: `finance_expenses`

| Turn | User Message |
|------|--------------|
| 1 | My monthly budget is $6,000. Housing costs $2,000 (rent), utilities $300, groceries $400. |
| 2 | Transportation is $400/month (car payment + insurance). Entertainment and dining out total $600. |
| 3 | I allocate $1,000/month for savings and $700 for personal care, insurance, and miscellaneous expenses. |
| 4 | Last month I overspent on dining out - spent $850 instead of $600. I also had an unexpected car repair that cost $450. |
| 5 | What's my monthly rent? (Recall test) |
| 6 | How much did I overspend on dining last month? (Recall test) |
| 7 | Give me a complete breakdown of my monthly budget and last month's actual spending. (Comprehensive recall test) |

### Scenario 16: Risk Assessment and Planning
**Namespace**: `finance_risk`

| Turn | User Message |
|------|--------------|
| 1 | I'm a 40-year-old professional with stable income of $180,000/year and relatively low expense obligations. |
| 2 | My risk tolerance is medium-to-high given my time horizon to retirement (25 years) and lack of dependents. |
| 3 | I have 6 months of emergency fund saved, so I'm well-protected against job loss or unexpected expenses. |
| 4 | My main financial risks are: market volatility, potential income disruption from industry changes, and sequence-of-returns risk as I near retirement. |
| 5 | What is my current age and time to retirement? (Recall test) |
| 6 | What emergency fund do I have in place? (Recall test) |
| 7 | Summarize my financial profile, risk tolerance, and identified risks. (Comprehensive recall test) |

## Code Generation (2 scenarios)

### Scenario 17: Architecture Design Discussion
**Namespace**: `code_architecture`

| Turn | User Message |
|------|--------------|
| 1 | I'm architecting a new microservices system for an e-commerce platform. The main services are: User Auth, Product Catalog, Shopping Cart, and Order Processing. |
| 2 | Each service has its own PostgreSQL database to ensure loose coupling. Services communicate via REST APIs and async message queues (RabbitMQ). |
| 3 | I'm using Docker for containerization and Kubernetes for orchestration. We'll have separate dev, staging, and production clusters. |
| 4 | For monitoring, we're implementing Prometheus for metrics and ELK stack for centralized logging. Each service has health check endpoints. |
| 5 | How many microservices am I building? (Recall test) |
| 6 | What database strategy am I using? (Recall test) |
| 7 | Describe my complete system architecture including services, databases, communication, and monitoring. (Comprehensive recall test) |

### Scenario 18: Bug Fixing and Debugging
**Namespace**: `code_bugs`

| Turn | User Message |
|------|--------------|
| 1 | I'm debugging a memory leak in my Node.js application. The heap size grows from 50MB to 500MB over 24 hours without dropping. |
| 2 | I've identified that the issue occurs in the data caching layer - specifically in the Redis client connection handling. Connections aren't being properly closed. |
| 3 | The bug manifests when the application processes more than 1,000 requests/minute. Under normal load, it's not noticeable. |
| 4 | I've fixed it by implementing proper connection pooling and adding a 5-minute timeout for idle connections. Code review is scheduled for tomorrow. |
| 5 | What service is experiencing the memory leak? (Recall test) |
| 6 | How much does heap size grow over 24 hours? (Recall test) |
| 7 | Summarize the bug, its cause, and my solution. (Comprehensive recall test) |

## Multimodal (2 scenarios)

### Scenario 19: Image-Based Visual Analysis
**Namespace**: `multi_images`

| Turn | User Message |
|------|--------------|
| 1 | I'm analyzing a series of satellite images of the Amazon rainforest taken over 10 years (2014-2024). |
| 2 | The images show significant deforestation in three regions: Northern Brazil, Peru border region, and Southern Brazil. I estimate 15-20% forest loss overall. |
| 3 | The most alarming trend is in Northern Brazil where deforestation increased 40% year-over-year from 2022-2024. |
| 4 | I'm creating a report with before/after image pairs to illustrate the changes. The data will be presented to environmental organizations next month. |
| 5 | What time period am I analyzing? (Recall test) |
| 6 | Which region shows the most alarming trend? (Recall test) |
| 7 | Summarize my satellite image analysis including regions, deforestation rates, and trends. (Comprehensive recall test) |

### Scenario 20: Project Documentation with Mixed Content
**Namespace**: `multi_project`

| Turn | User Message |
|------|--------------|
| 1 | I'm documenting a climate modeling software project. The codebase is 50,000 lines of Python with modules for data ingestion, climate simulation, and visualization. |
| 2 | The project includes 200+ scientific papers as references, covering topics from atmospheric dynamics to ocean circulation modeling. |
| 3 | I've created architectural diagrams showing data flow from raw weather data inputs through simulation engines to output dashboards. |
| 4 | I'm also collecting testimonials from 15 climate scientists who use the software. They report that it's 3x faster than competing tools and very user-friendly. |
| 5 | How many lines of code are in the project? (Recall test) |
| 6 | How many scientific papers are referenced? (Recall test) |
| 7 | Describe my climate modeling project including code size, references, architecture, and user feedback. (Comprehensive recall test) |

---

## Testing Instructions

These scenarios are designed to evaluate Ucotron's:
- **Context Retention**: Can the system remember facts across multiple turns?
- **Contradiction Resolution**: Does it detect and resolve conflicting information (scenarios 2, 4, 6)?
- **Recall Accuracy**: Can it accurately answer questions about previously stated information?
- **Comprehensive Understanding**: Can it synthesize information across multiple turns into coherent summaries?
- **Multilingual Processing**: Scenarios 12 and others mix Spanish/English (marked with parenthetical notes).

Each scenario's recall tests (turns 5+) should validate that the system has correctly ingested and can retrieve the information provided in earlier turns.
