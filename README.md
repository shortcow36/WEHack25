## Inspiration
Small businesses face countless hidden risks, from signing unfavorable leases to partnering with the wrong people. These decisions often have long-term consequences, yet most small business owners lack the tools to evaluate those risks, especially when considering shared or co-ventured spaces.

We wanted to build something that helps real people, not just investors, make smarter, safer decisions in moments that can define the future of their business.

## What it does
Co-SQRD is Co-venturing and risk analysis made easy, an AI-powered tool that helps small business owners:
- Match with the right co-venturer based on trust, compatibility, and fit
- Evaluate location risk scores using property characteristics such as crime rate, foot traffic, and appreciation rate
- Understand operational fit and potential conflicts before signing a lease

## How we built it
- Frontend: Developed a clean, user-centered interface using Streamlit
- Backend: Integrated a Python-based TensorFlow model to calculate risk scores

## Challenges we ran into
- Real estate APIs were either paywalled or too broad, and no niche-free options were available for our use case
- We had to collect and clean real commercial addresses for our dataset manually due to the lack of publicly available datasets for commercial property risk analysis
- With limited time, we couldn't build or integrate a full machine-learning model for risk-scoring
- We faced limitations in customizing the interface due to Streamlit's restricted support for CSS styling

## Accomplishments that we're proud of
- Identified and tackled a real-world gap in risk assessment for commercial properties that directly impacts real people
- Developed a functional prototype with realistic, insightful data and a user-friendly interface
- Defined a clear product vision with strong future potential for scalability in the real estate space

## What we learned
- You don’t have to build everything to build something meaningful
- Focused execution, even with time limits, can make a niche idea come alive
- Even simple features like foot traffic and nearby businesses can reveal powerful insights when framed well
- Limited data paired with strong design choices can power meaningful and functional demos

## What's next for Co-SQRD
- Add a call and messaging system powered by an AI agent
- Expand to more cities and add real-time business profile-matching
- Incorporate operational fit into the property and risk evaluation model
- Integrate with commercial real estate APIs for live listings and smarter scoring
