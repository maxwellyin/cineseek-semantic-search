import { Layers, Sparkles, Zap } from 'lucide-react';

function ProjectOverview() {
  return (
    <div className="overview-view">
      <section className="page-card">
        <div className="eyebrow overview-gap-sm">Project Overview</div>
        <h1 className="display-title overview-title">A retrieval-first search system wrapped in a usable product.</h1>
        <p className="body-copy overview-lede">
          CineSeek is built to show more than a model checkpoint. It combines dense retrieval, FAISS serving, optional
          LLM reranking, MCP tool exposure, and deployable infrastructure in one search experience.
        </p>
      </section>

      <section className="split-grid">
        <div className="page-card">
          <div className="eyebrow overview-gap-sm">What It Demonstrates</div>
          <h2 className="section-title">A retrieval-first stack with an optional agent layer.</h2>
          <div className="feature-grid">
            <FeatureCard
              icon={<Layers size={26} />}
              title="Dense Retrieval"
              copy="Queries and movies are mapped into a shared representation space so results can match meaning, not just keyword overlap."
            />
            <FeatureCard
              icon={<Zap size={26} />}
              title="Candidate Search"
              copy="FAISS provides low-latency nearest-neighbor retrieval over a movie corpus built from MovieLens and TMDB metadata."
            />
            <FeatureCard
              icon={<Sparkles size={26} />}
              title="LLM Reranking"
              copy="A configurable agent layer can rewrite vague queries, rerank candidates, and summarize the final result list."
            />
          </div>
        </div>

        <div className="page-card">
          <div className="eyebrow overview-gap-sm">System Flow</div>
          <h2 className="section-title">What happens after you search</h2>
          <div className="stack-grid">
            <FlowCard
              title="1. Retrieve"
              copy="The query is embedded and matched against a FAISS index of movie representations."
            />
            <FlowCard
              title="2. Rerank"
              copy="The agent reviews a broader candidate set, improves ordering, and filters obviously weak matches."
            />
            <FlowCard
              title="3. Explain"
              copy="The final page includes a short summary so the result list feels inspectable rather than opaque."
            />
          </div>
        </div>
      </section>

      <section className="page-card">
        <div className="eyebrow overview-gap-sm">Recruiter Summary</div>
        <h2 className="section-title">An ML systems project, not just a movie site.</h2>
        <div className="metric-grid">
          <div className="metric-card">
            <div className="metric-value">~28k</div>
            <div className="metric-label">Real movie search queries from MSRD</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">9,691</div>
            <div className="metric-label">Indexed candidate movies in the retrieval corpus</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">MCP</div>
            <div className="metric-label">Retrieval engine exposed as a reusable search tool</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">Docker + CI/CD</div>
            <div className="metric-label">Containerized deployment, health checks, and automated publishing</div>
          </div>
        </div>
      </section>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .overview-view {
              display: flex;
              flex-direction: column;
              gap: 20px;
            }

            .overview-title {
              max-width: none;
            }

            .overview-lede {
              max-width: 82ch;
            }

            .overview-gap-sm {
              margin-bottom: 12px;
            }

            .split-grid {
              display: grid;
              grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
              gap: 20px;
            }

            .feature-grid {
              display: grid;
              grid-template-columns: repeat(3, minmax(0, 1fr));
              gap: 16px;
              margin-top: 22px;
            }

            .stack-grid {
              display: grid;
              gap: 14px;
              margin-top: 22px;
            }

            .feature-copy {
              margin: 0;
              color: var(--muted);
              line-height: 1.65;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            }

            .metric-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(0, 1fr));
              gap: 14px;
              margin-top: 22px;
            }

            .metric-card {
              background: rgba(255, 255, 255, 0.76);
              border: 1px solid var(--line);
              border-radius: 20px;
              padding: 18px;
            }

            .metric-value {
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 1.2rem;
              font-weight: 800;
              color: var(--accent-dark);
            }

            .metric-label {
              margin-top: 8px;
              color: var(--muted);
              line-height: 1.6;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
            }

            @media (max-width: 980px) {
              .split-grid {
                grid-template-columns: 1fr;
              }
            }

            @media (max-width: 720px) {
              .feature-grid,
              .metric-grid {
                grid-template-columns: 1fr;
              }
            }
          `,
        }}
      />
    </div>
  );
}

function FeatureCard({ icon, title, copy }) {
  return (
    <div className="feature-card">
      <div className="feature-icon">{icon}</div>
      <h3 className="inter feature-title">{title}</h3>
      <p className="feature-copy">{copy}</p>
      <style
        dangerouslySetInnerHTML={{
          __html: `
            .feature-card {
              background: rgba(255, 255, 255, 0.76);
              border: 1px solid var(--line);
              border-radius: 22px;
              padding: 20px;
            }

            .feature-icon {
              color: var(--accent);
              margin-bottom: 12px;
            }

            .feature-title {
              margin: 0 0 10px;
              font-size: 1.08rem;
            }
          `,
        }}
      />
    </div>
  );
}

function FlowCard({ title, copy }) {
  return (
    <div className="flow-card">
      <h3 className="inter flow-title">{title}</h3>
      <p className="feature-copy">{copy}</p>
      <style
        dangerouslySetInnerHTML={{
          __html: `
            .flow-card {
              background: rgba(255, 255, 255, 0.76);
              border: 1px solid var(--line);
              border-radius: 22px;
              padding: 20px;
            }

            .flow-title {
              margin: 0 0 8px;
              font-size: 1.04rem;
            }
          `,
        }}
      />
    </div>
  );
}

export default ProjectOverview;
