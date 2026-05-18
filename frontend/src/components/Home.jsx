import { Search } from 'lucide-react';
import { useRef, useState } from 'react';

const HOME_EXAMPLES = [
  'Mind-bending movies like Inception but darker',
  'Emotional sci-fi about space, family, and sacrifice',
  'Underrated psychological thrillers',
];

function Home({ defaultQuery, agentAvailable, onSearch }) {
  const textareaRef = useRef(null);
  const [loading, setLoading] = useState(false);

  const submitSearch = (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const text = String(formData.get('text') || '');
    const useAgent = agentAvailable ? formData.get('use_agent') === '1' : false;
    setLoading(true);
    onSearch({ text, useAgent });
  };

  const fillQuery = (query) => {
    if (!textareaRef.current) {
      return;
    }
    textareaRef.current.value = query;
    textareaRef.current.focus();
  };

  return (
    <div className="home-view">
      <section className="page-card hero-card">
        <div className="eyebrow home-gap-sm">Search Interface</div>
        <h1 className="display-title hero-title">CineSeek — LLM-powered semantic movie search.</h1>
        <p className="body-copy hero-lede">
          Search movies using natural language, not keywords. CineSeek combines dense retrieval, FAISS candidate
          search, and LLM reranking to better match user intent on fuzzy movie queries.
        </p>

        <form className="search-stack home-gap-md" onSubmit={submitSearch}>
          <label className="mini-label home-gap-xs" htmlFor="home-query-box">
            Try a query
          </label>
          <textarea
            id="home-query-box"
            name="text"
            className="query-box"
            placeholder={`Try: "${defaultQuery}"`}
            rows={1}
            ref={textareaRef}
          />

          <div className="hero-actions">
            <div>
              <label className="toggle-row">
                <input type="checkbox" name="use_agent" value="1" defaultChecked={agentAvailable} disabled={!agentAvailable} />
                <span>Use LLM agent for reranking and explanation</span>
              </label>
              <div className="mini-label hero-note">Results are powered by semantic retrieval + LLM reranking.</div>
            </div>
            <button type="submit" className="primary-btn">
              <Search size={18} />
              <span>Search Now</span>
            </button>
          </div>

          <div className="home-gap-md">
            <div className="mini-label home-gap-xs">Try one of these</div>
            <div className="chip-row">
              {HOME_EXAMPLES.map((query) => (
                <button
                  key={query}
                  type="button"
                  className="query-chip"
                  onClick={() => fillQuery(query)}
                >
                  {query}
                </button>
              ))}
            </div>
          </div>

          {loading && (
            <div className="loading-panel">
              <div className="loading-row">
                <div className="loading-orb" aria-hidden="true"></div>
                <div>
                  <div className="mini-label home-gap-xs">Working</div>
                  <div className="body-copy loading-copy">
                    Please wait while CineSeek retrieves candidates, reranks them, and prepares the final results.
                  </div>
                </div>
              </div>
            </div>
          )}
        </form>
      </section>

      <style
        dangerouslySetInnerHTML={{
          __html: `
            .home-view {
              display: flex;
              flex-direction: column;
              gap: 24px;
            }

            .hero-card {
              padding: 40px 48px;
            }

            .hero-title {
              font-size: clamp(2rem, 3.8vw, 2.85rem);
              line-height: 1.12;
              max-width: 28ch;
              margin-top: 0;
              margin-bottom: 18px;
              letter-spacing: -0.02em;
            }

            .hero-lede {
              max-width: 68ch;
              font-size: 1.05rem;
              line-height: 1.65;
              opacity: 0.85;
              margin-bottom: 24px;
            }

            .home-gap-xs {
              margin-bottom: 8px;
            }

            .home-gap-sm {
              margin-bottom: 12px;
            }

            .home-gap-md {
              margin-top: 24px;
            }

            .search-stack {
              display: flex;
              flex-direction: column;
            }

            .query-box {
              width: 100%;
              min-height: 56px;
              height: 56px;
              border-radius: 16px;
              border: 1px solid rgba(24, 34, 47, 0.05);
              background: rgba(255, 255, 255, 0.88);
              padding: 16px 20px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.98rem;
              line-height: 1.5;
              box-shadow: 
                0 2px 8px rgba(24, 34, 47, 0.02),
                inset 0 1px 2px rgba(24, 34, 47, 0.03);
              resize: none;
              overflow-y: hidden;
              transition: min-height var(--transition-smooth), height var(--transition-smooth), border-color var(--transition-smooth), box-shadow var(--transition-smooth), background-color var(--transition-smooth);
            }

            .query-box::placeholder {
              color: var(--muted);
              opacity: 0.5;
            }

            .query-box:hover {
              border-color: rgba(198, 93, 46, 0.2);
              box-shadow: 
                0 4px 12px rgba(24, 34, 47, 0.03),
                inset 0 1px 2px rgba(24, 34, 47, 0.02);
            }

            .query-box:focus {
              outline: none;
              min-height: 110px;
              height: 110px;
              background: #ffffff;
              border-color: rgba(198, 93, 46, 0.35);
              box-shadow: 
                0 12px 30px rgba(198, 93, 46, 0.06), 
                0 0 0 4px rgba(198, 93, 46, 0.08);
              overflow-y: auto;
            }

            .hero-actions {
              display: flex;
              align-items: center;
              justify-content: space-between;
              gap: 16px;
              margin-top: 20px;
            }

            .toggle-row {
              display: inline-flex;
              align-items: center;
              gap: 10px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              color: var(--ink);
              opacity: 0.85;
              font-weight: 500;
              cursor: pointer;
            }

            .toggle-row input {
              margin: 0;
              cursor: pointer;
              width: 16px;
              height: 16px;
              accent-color: var(--accent);
            }

            .hero-note {
              margin-top: 6px;
              opacity: 0.7;
            }

            .primary-btn {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              gap: 10px;
              background: #C96B3B;
              color: #fff;
              border-radius: 999px;
              padding: 13px 24px;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 700;
              font-size: 0.92rem;
              box-shadow: 0 4px 14px rgba(201, 107, 59, 0.15);
              transition: transform var(--transition-smooth), box-shadow var(--transition-smooth), background-color var(--transition-smooth);
            }

            .primary-btn:hover {
              background: #b55a2d;
              transform: translateY(-1.5px);
              box-shadow: 
                0 8px 24px rgba(201, 107, 59, 0.22), 
                0 0 0 4px rgba(201, 107, 59, 0.08);
            }

            .primary-btn:active {
              background: #9d4c23;
              transform: translateY(0);
              box-shadow: 0 3px 10px rgba(201, 107, 59, 0.12);
            }

            .chip-row {
              display: flex;
              flex-wrap: wrap;
              gap: 10px;
            }

            .loading-panel {
              margin-top: 24px;
              border-radius: 24px;
              border: 1px solid var(--line);
              background: rgba(255, 255, 255, 0.72);
              padding: 20px;
              backdrop-filter: blur(8px);
            }

            .loading-row {
              display: flex;
              align-items: center;
              gap: 14px;
            }

            .loading-orb {
              width: 1.1rem;
              height: auto;
              aspect-ratio: 1 / 1;
              border-radius: 999px;
              border: 2px solid rgba(198, 93, 46, 0.22);
              border-top-color: var(--accent);
              animation: spin 1s linear infinite;
              flex: 0 0 auto;
            }

            .loading-copy {
              margin: 0;
              opacity: 0.8;
            }

            .query-chip {
              border: 1px solid rgba(24, 34, 47, 0.08);
              border-radius: 999px;
              background: rgba(250, 248, 245, 0.75);
              color: var(--ink);
              padding: 10px 16px;
              text-align: left;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-size: 0.9rem;
              font-weight: 500;
              transition: background-color var(--transition-smooth), border-color var(--transition-smooth), transform var(--transition-smooth), box-shadow var(--transition-smooth);
            }

            .query-chip:hover {
              background-color: #ffffff;
              border-color: rgba(198, 93, 46, 0.3);
              transform: translateY(-2px);
              box-shadow: 0 4px 12px rgba(24, 34, 47, 0.04);
            }

            .query-chip:active {
              transform: translateY(0);
            }

            .mini-label {
              text-transform: uppercase;
              letter-spacing: 0.08em;
              font-size: 0.7rem;
              font-family: Inter, ui-sans-serif, system-ui, sans-serif;
              font-weight: 700;
              color: var(--muted);
              opacity: 0.8;
            }

            @media (max-width: 768px) {
              .hero-card {
                padding: 30px 24px;
              }
              .hero-actions {
                flex-direction: column;
                align-items: stretch;
                gap: 16px;
              }
              .primary-btn {
                width: 100%;
              }
            }

            @keyframes spin {
              from { transform: rotate(0deg); }
              to { transform: rotate(360deg); }
            }
          `,
        }}
      />
    </div>
  );
}

export default Home;
