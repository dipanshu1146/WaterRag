import { useEffect, useState } from "react";
import "./App.css";

const API = (
  import.meta.env.BACKEND_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const examples = [
  { icon: "◉", text: "Latest outlet BOD, COD and TSS for GRN-STP-011" },
  { icon: "⌕", text: "Compare outlet results across Panipat STPs" },
  { icon: "↗", text: "Total treatment capacity in Panipat?" }
];

function Icon({ name, size = 18 }) {
  const paths = {
    search: <><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4 4"/></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21z"/><path d="M4 5.5v13A2.5 2.5 0 0 1 6.5 16H20"/></>,
    upload: <><path d="M12 16V4m-4 4 4-4 4 4"/><path d="M5 14v5h14v-5"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="m19.4 15 .1.1 1.2 2.1-2 2-2.1-1.2-.2.1a8 8 0 0 1-1.7.7l-.7 2.3h-2.8l-.7-2.3a8 8 0 0 1-1.7-.7l-.2-.1-2.1 1.2-2-2 1.2-2.1-.1-.2a8 8 0 0 1-.7-1.7L2.5 12v-2.8l2.3-.7a8 8 0 0 1 .7-1.7l.1-.2-1.2-2.1 2-2 2.1 1.2.2-.1a8 8 0 0 1 1.7-.7L11.1.6h2.8l.7 2.3a8 8 0 0 1 1.7.7l.2.1 2.1-1.2 2 2-1.2 2.1.1.2a8 8 0 0 1 .7 1.7l2.3.7V12l-2.3.7a8 8 0 0 1-.7 1.7z" transform="translate(0 1) scale(.9)"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    file: <><path d="M13 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10z"/><path d="M13 3v7h7"/></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
    refresh: <><path d="M20 7v5h-5M4 17v-5h5"/><path d="M5.5 9a7 7 0 0 1 11.6-2L20 12M4 12l2.9 5a7 7 0 0 0 11.6-2"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [documents, setDocuments] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [deletingFile, setDeletingFile] = useState("");
  const [activeView, setActiveView] = useState("ask");
  const [guideOpen, setGuideOpen] = useState(false);

  async function loadDocuments() {
    setDocumentsLoading(true);
    setDocumentsError("");
    try {
      const response = await fetch(`${API}/api/documents`);
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not load documents.");
      setDocuments(data.documents || []);
    } catch (err) {
      setDocumentsError(err.message || "Could not load documents.");
    } finally { setDocumentsLoading(false); }
  }

  async function deleteDocument(doc) {
    if (!window.confirm(`Delete "${doc.original_filename}" and its indexed records?`)) return;
    setDeletingFile(doc.stored_filename); setDocumentsError("");
    try {
      const response = await fetch(`${API}/api/documents/${encodeURIComponent(doc.stored_filename)}`, { method: "DELETE" });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not delete document.");
      setDocuments((current) => current.filter((item) => item.stored_filename !== doc.stored_filename));
      setUploadMessage(`${doc.original_filename} removed from your library.`);
    } catch (err) { setDocumentsError(err.message || "Could not delete document."); }
    finally { setDeletingFile(""); }
  }

  async function reindexDocument(doc) {
    setDeletingFile(doc.stored_filename); setDocumentsError(""); setUploadMessage("");
    try {
      const response = await fetch(`${API}/api/documents/${encodeURIComponent(doc.stored_filename)}/reindex`, { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.message || "Could not re-index document.");
      setUploadMessage(`${doc.original_filename} re-indexed · ${data.indexed_records} records`);
      await loadDocuments();
    } catch (err) { setDocumentsError(err.message || "Could not re-index document."); }
    finally { setDeletingFile(""); }
  }

  function fileLabel(name) {
    const match = /\.([A-Za-z0-9]+)$/.exec(name || "");
    return match ? match[1].toUpperCase().slice(0, 4) : "FILE";
  }
  function formatFileSize(bytes = 0) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  useEffect(() => { loadDocuments(); }, []);
  useEffect(() => {
    if (!guideOpen) return undefined;
    const closeOnEscape = (event) => { if (event.key === "Escape") setGuideOpen(false); };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [guideOpen]);
  useEffect(() => {
    fetch(`${API}/api/health`).then((r) => { if (!r.ok) throw new Error(); return r.json(); })
      .then(() => setBackendStatus("Connected")).catch(() => setBackendStatus("Disconnected"));
  }, []);

  async function uploadDocument(e) {
    e.preventDefault(); if (!selectedFile || uploading) return;
    setUploading(true); setUploadMessage(""); setUploadError("");
    try {
      const formData = new FormData(); formData.append("file", selectedFile);
      const response = await fetch(`${API}/api/documents/upload`, { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.message || `Upload failed (${response.status})`);
      if (!data.indexed_records || data.indexed_records <= 0) throw new Error(`${data.original_filename} was saved but no content could be indexed.`);
      setUploadMessage(`${data.original_filename} is ready to search · ${data.indexed_records} records indexed`);
      await loadDocuments(); setSelectedFile(null); e.target.reset();
    } catch (err) { setUploadError(err.message || "Could not upload the file."); loadDocuments(); }
    finally { setUploading(false); }
  }

  async function askQuestion(e) {
    e?.preventDefault(); if (!question.trim() || loading) return;
    setActiveView("ask"); setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch(`${API}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: question.trim() }) });
      if (!response.ok) throw new Error(`Request failed (${response.status})`);
      setResult(await response.json());
    } catch (err) { setError(`${err.message}. Check that the backend URL is configured and reachable.`); }
    finally { setLoading(false); }
  }

  const indexedCount = documents.filter((doc) => doc.status === "indexed").length;
  const showLibrary = activeView === "library";

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" onClick={() => setActiveView("ask")}>
          <span className="brand-mark"><span /></span>
          <span className="brand-copy"><strong>Clearwater</strong><small>WATER INTELLIGENCE</small></span>
        </a>
        <div className="workspace-picker"><span className="workspace-avatar">W</span><span><strong>Water quality</strong><small>Workspace</small></span><span className="chevron">⌄</span></div>
        <div className="nav-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <button className={!showLibrary ? "nav-item active" : "nav-item"} onClick={() => setActiveView("ask")}><Icon name="search"/><span>Ask your data</span><kbd>⌘ K</kbd></button>
          <button className={showLibrary ? "nav-item active" : "nav-item"} onClick={() => setActiveView("library")}><Icon name="book"/><span>Document library</span><b className="nav-count">{documents.length}</b></button>
        </nav>
        <div className="sidebar-bottom">
          <button type="button" className="sidebar-help" onClick={() => setGuideOpen(true)} aria-haspopup="dialog"><span className="help-icon">?</span><span><strong>Need a hand?</strong><small>Learn how to use Clearwater</small></span><Icon name="arrow" size={15}/></button>
          <div className="user-profile"><div className="user-avatar">AP</div><span><strong>Workspace admin</strong><small>Water Research Team</small></span><span className="more">···</span></div>
        </div>
      </aside>

      <div className="app-main">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><span className="crumb-sep">/</span><strong>{showLibrary ? "Document library" : "Ask your data"}</strong></div>
          <div className="topbar-actions"><span className={`connection ${backendStatus === "Connected" ? "online" : "offline"}`}><i />{backendStatus === "Connected" ? "All systems operational" : backendStatus === "Checking..." ? "Connecting to API" : "API unavailable"}</span><button className="avatar-small" aria-label="Account">AP</button></div>
        </header>

        <main className="page-content">
          {showLibrary ? <>
            <div className="page-heading library-heading"><div><div className="eyebrow">KNOWLEDGE BASE</div><h1>Your document library</h1><p>Manage the reports and datasets Clearwater uses to answer questions.</p></div><button className="primary-button" onClick={() => document.getElementById("document-file")?.click()}><Icon name="plus" size={17}/> Add documents</button></div>
            <div className="library-toolbar"><div className="library-stats"><strong>{documents.length}</strong> documents <span>·</span> <strong>{indexedCount}</strong> indexed</div><button className="quiet-button" onClick={loadDocuments} disabled={documentsLoading}><Icon name="refresh" size={16}/>{documentsLoading ? "Refreshing" : "Refresh"}</button></div>
            {documentsError && <div className="error-panel" role="alert">{documentsError}</div>}
            {uploadMessage && <div className="success-panel" role="status">{uploadMessage}</div>}
            <section className="library-table">
              <div className="library-table-head"><span>DOCUMENT</span><span>STATUS</span><span>SIZE</span><span>RECORDS</span><span /></div>
              {documentsLoading && documents.length === 0 && <div className="library-empty">Loading your documents…</div>}
              {!documentsLoading && documents.length === 0 && !documentsError && <div className="library-empty"><div className="empty-mark"><Icon name="book" size={21}/></div><strong>Your library is ready for its first document</strong><span>Add a report or dataset to start asking questions.</span><button className="primary-button" onClick={() => document.getElementById("document-file")?.click()}><Icon name="plus" size={17}/> Add a document</button></div>}
              {documents.map((doc) => <div className="library-row" key={doc.stored_filename}><div className="library-name"><span className="file-badge">{fileLabel(doc.original_filename)}</span><span><strong title={doc.original_filename}>{doc.original_filename}</strong><small>Added to this workspace</small></span></div><div><span className={`status-pill ${doc.status === "indexed" ? "pill-ready" : "pill-pending"}`}><i/>{doc.status === "indexed" ? "Indexed" : "Needs indexing"}</span></div><span className="table-muted">{formatFileSize(doc.size_bytes)}</span><span className="table-muted">{doc.indexed_records ?? 0}</span><div className="row-actions">{doc.status !== "indexed" && <button className="text-action" onClick={() => reindexDocument(doc)} disabled={Boolean(deletingFile)}>{deletingFile === doc.stored_filename ? "Working…" : "Re-index"}</button>}<button className="text-action danger" onClick={() => deleteDocument(doc)} disabled={Boolean(deletingFile)} aria-label={`Delete ${doc.original_filename}`}>{deletingFile === doc.stored_filename ? "Working…" : "Remove"}</button></div></div>)}
            </section>
            <section className="upload-card"><div className="upload-card-icon"><Icon name="upload" size={20}/></div><div><strong>Add another source</strong><p>PDF, Excel, CSV, DOCX or TXT · up to 25 MB</p></div><form className="inline-upload" onSubmit={uploadDocument}><input id="document-file" type="file" accept=".pdf,.xlsx,.csv,.docx,.txt" onChange={(e) => {setSelectedFile(e.target.files?.[0] || null);setUploadMessage("");setUploadError("");}}/><span className="selected-file">{selectedFile?.name || ""}</span><button type="submit" className="secondary-button" disabled={!selectedFile || uploading}>{uploading ? "Uploading…" : "Choose file"}</button></form></section>
            {uploadError && <div className="error-panel" role="alert">{uploadError}</div>}
          </> : <>
            <div className="page-heading"><div><div className="eyebrow">WATER QUALITY WORKSPACE</div><h1>Good morning. What would you like to know?</h1><p>Ask a question across your wastewater reports and get an answer grounded in your data.</p></div><span className="date-chip">✦ &nbsp; Grounded in your sources</span></div>
            <div className="workspace-grid">
              <section className="ask-column">
                <form className="composer" onSubmit={askQuestion}>
                  <div className="composer-label"><span className="sparkle">✳</span><span>ASK CLEARWATER</span><span className="scope-chip"><Icon name="book" size={14}/> All documents <span>⌄</span></span></div>
                  <textarea value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask about treatment performance, sample results, or a report…" rows={4} aria-label="Your question" />
                  <div className="composer-bottom"><span className="composer-hint">Answers include source references for easy verification.</span><button className="ask-button" type="submit" disabled={loading || !question.trim()}>{loading ? "Searching…" : <>Ask question <Icon name="arrow" size={16}/></>}</button></div>
                </form>
                {loading && <div className="loading-card"><span className="loader"/><span><strong>Searching your documents</strong><small>Finding relevant passages and checking sources…</small></span></div>}
                {error && <div className="error-panel" role="alert">{error}</div>}
                {result && <section className="answer-card"><div className="answer-header"><span className="answer-badge">✳</span><div><strong>Answer</strong><small>Generated from your indexed documents</small></div><button className="icon-button" onClick={() => setResult(null)} aria-label="Close answer"><Icon name="close" size={17}/></button></div><p className="answer-text">{result.answer || "No answer was returned."}</p>{result.summary && <div className="answer-summary"><span>RETRIEVAL SUMMARY</span><strong>{result.summary.records_count ?? "—"} records found</strong>{result.summary.sample_results_found === false && <small>No dated sample results found</small>}</div>}
                  {Array.isArray(result.records) && result.records.some((record) => record.stp_id || record.stp_name || record.sample_section) && <div className="table-wrap"><div className="section-title"><strong>Retrieved records</strong><span>{result.records.length} results</span></div><table><thead><tr><th>STP ID</th><th>Sample type</th><th>Date</th><th>BOD</th><th>COD</th><th>TSS</th><th>pH</th></tr></thead><tbody>{result.records.map((record,index)=><tr key={`${record.stp_id || "record"}-${index}`}><td>{record.stp_id || record.stp_name || "—"}</td><td>{record.sample_section || "—"}</td><td>{record.sample_date || "—"}</td><td>{record.bod ?? "—"}</td><td>{record.cod ?? "—"}</td><td>{record.tss ?? "—"}</td><td>{record.ph ?? "—"}</td></tr>)}</tbody></table></div>}
                  {Array.isArray(result.sources) && result.sources.length > 0 && <div className="sources"><div className="section-title"><strong>Sources</strong><span>{result.sources.length} references</span></div>{result.sources.map((source,index)=><details key={source.source_id ?? index}><summary><span className="source-file"><Icon name="file" size={16}/></span><span><strong>{source.document || source.source || `Source ${index + 1}`}</strong><small>{source.page && source.page !== "N/A" ? `Page ${source.page}` : "Document passage"}{source.table_number && source.table_number !== "N/A" ? ` · Table ${source.table_number}` : ""}</small></span><span className="source-open">⌄</span></summary><pre>{source.content || source.text || "Source content not provided."}</pre></details>)}</div>}
                </section>}
                {!result && !loading && <div className="privacy-note"><span className="lock-icon">⌑</span> Your answers are generated only from documents in this workspace.</div>}
              </section>
              <aside className="right-rail">
                <section className="rail-card resources-card"><div className="rail-card-heading"><span className="rail-icon teal"><Icon name="book" size={17}/></span><span><strong>Workspace resources</strong><small>Sources used to ground answers</small></span><span className="resource-total">{documents.length}</span></div><div className="resource-overview"><strong>{indexedCount}<span> of {documents.length} indexed</span></strong><div className="progress-track"><span style={{width: `${documents.length ? (indexedCount / documents.length) * 100 : 0}%`}}/></div><div className="progress-caption"><span>Searchable resources</span><span>{documents.length - indexedCount} need attention</span></div></div><div className="resource-list">{documents.length ? documents.map((doc)=><div className="resource-item" key={doc.stored_filename}><span className="recent-file">{fileLabel(doc.original_filename)}</span><span className="resource-name"><strong title={doc.original_filename}>{doc.original_filename}</strong><small>{doc.status === "indexed" ? `${doc.indexed_records} records · Indexed` : "Needs indexing"}</small></span>{doc.status !== "indexed" && <button className="resource-reindex" onClick={() => reindexDocument(doc)} disabled={Boolean(deletingFile)} aria-label={`Re-index ${doc.original_filename}`}>↻</button>}<button className="resource-remove" onClick={() => deleteDocument(doc)} disabled={Boolean(deletingFile)} aria-label={`Remove ${doc.original_filename}`} title="Remove resource">×</button></div>) : <p className="resources-empty">No resources yet. Add reports or datasets to ask questions about your water data.</p>}</div><button className="resource-add" onClick={() => setActiveView("library")}><Icon name="plus" size={15}/> Add resources</button></section>
                <section className="rail-card suggestions-card"><div className="rail-card-heading"><span className="rail-icon lavender"><Icon name="search" size={17}/></span><span><strong>Try asking</strong><small>Suggested questions</small></span></div><div className="suggestion-list">{examples.map((example)=><button className="suggestion-item" key={example.text} onClick={() => {setQuestion(example.text);document.querySelector(".composer textarea")?.focus();}}><span className="suggestion-symbol">{example.icon}</span><span>{example.text}</span><Icon name="arrow" size={14}/></button>)}</div></section>
              </aside>
            </div>
          </>}
          <footer className="page-footer"><span>Clearwater <span className="footer-dot">·</span> Wastewater intelligence</span><span className="footer-state"><i/>{backendStatus === "Connected" ? "Connected securely" : backendStatus === "Checking..." ? "Checking connection" : "Backend disconnected"}</span></footer>
        </main>
      </div>
      {guideOpen && <div className="guide-overlay" onMouseDown={(event) => { if (event.target === event.currentTarget) setGuideOpen(false); }}><section className="guide-dialog" role="dialog" aria-modal="true" aria-labelledby="guide-title" aria-describedby="guide-intro"><header className="guide-header"><div><div className="guide-eyebrow">CLEARWATER QUICK GUIDE</div><h2 id="guide-title">Your water data, easier to explore.</h2><p id="guide-intro">Clearwater searches the reports you add, then gives you a concise answer with the records and sources behind it.</p></div><button className="guide-close" type="button" onClick={() => setGuideOpen(false)} aria-label="Close guide"><Icon name="close" size={18}/></button></header>
          <div className="guide-flow-wrap"><div className="guide-flow-title"><strong>How a grounded answer works</strong><span>Four steps, from report to reference</span></div><div className="guide-flow-scroll"><svg className="guide-flow" viewBox="0 0 800 152" role="img" aria-labelledby="clearwater-flow-title clearwater-flow-desc"><title id="clearwater-flow-title">How Clearwater answers a question</title><desc id="clearwater-flow-desc">Add documents, retrieve matching passages, generate an answer grounded in those passages, then verify the cited source.</desc><defs><marker id="guide-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1 6 4 1 7" fill="none" stroke="#82978e" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/></marker></defs><path d="M184 76H213M384 76H413M584 76H613" fill="none" stroke="#82978e" strokeWidth="1.2" markerEnd="url(#guide-arrow)"/><g><rect x="16" y="28" width="168" height="96" rx="10" fill="#fff" stroke="#cfe5db"/><circle cx="40" cy="51" r="12" fill="#e9f5f0"/><text x="40" y="55" textAnchor="middle" fontSize="11" fontWeight="700" fill="#317d68">1</text><text x="32" y="82" fontSize="12" fontWeight="700" fill="#30483e">Add resources</text><text x="32" y="101" fontSize="9" fill="#72857b">Upload a report or dataset</text></g><g><rect x="216" y="28" width="168" height="96" rx="10" fill="#fff" stroke="#cfe5db"/><circle cx="240" cy="51" r="12" fill="#e9f5f0"/><text x="240" y="55" textAnchor="middle" fontSize="11" fontWeight="700" fill="#317d68">2</text><text x="232" y="82" fontSize="12" fontWeight="700" fill="#30483e">Find relevant data</text><text x="232" y="101" fontSize="9" fill="#72857b">Search matching passages</text></g><g><rect x="416" y="28" width="168" height="96" rx="10" fill="#fff" stroke="#cfe5db"/><circle cx="440" cy="51" r="12" fill="#e9f5f0"/><text x="440" y="55" textAnchor="middle" fontSize="11" fontWeight="700" fill="#317d68">3</text><text x="432" y="82" fontSize="12" fontWeight="700" fill="#30483e">Get an answer</text><text x="432" y="101" fontSize="9" fill="#72857b">Written from retrieved content</text></g><g><rect x="616" y="28" width="168" height="96" rx="10" fill="#fff" stroke="#cfe5db"/><circle cx="640" cy="51" r="12" fill="#e9f5f0"/><text x="640" y="55" textAnchor="middle" fontSize="11" fontWeight="700" fill="#317d68">4</text><text x="632" y="82" fontSize="12" fontWeight="700" fill="#30483e">Check the sources</text><text x="632" y="101" fontSize="9" fill="#72857b">Open citations to verify</text></g></svg></div></div>
          <div className="guide-sections"><article className="guide-section"><span className="guide-section-icon">01</span><div><h3>Start with your resources</h3><p>Choose <strong>Document library</strong> or <strong>Add resources</strong> to upload PDF, XLSX, CSV, DOCX, or TXT files up to 25 MB. Indexed resources are searchable. Re-index a resource if it needs to be processed again; remove it to delete it and its indexed records.</p></div></article><article className="guide-section"><span className="guide-section-icon">02</span><div><h3>Ask a specific question</h3><p>Include an STP ID, sample type, parameter, and date range when you know them. For example: <em>“Show the latest outlet BOD, COD and TSS results for GRN-STP-011.”</em> Select a suggested question to place it in the composer, then choose <strong>Ask question</strong>.</p></div></article><article className="guide-section"><span className="guide-section-icon">03</span><div><h3>Review and verify</h3><p>Read the answer and any retrieved records. Expand a source to inspect its passage and page or table reference. If a source doesn’t support the answer, refine your question or check the uploaded document.</p></div></article></div>
          <div className="guide-controls"><strong>What the controls do</strong><div className="guide-control-list"><span><b>Ask your data</b> — return to the question workspace</span><span><b>Document library</b> — add, refresh, re-index, or remove files</span><span><b>Workspace resources</b> — review indexed status and remove a resource</span><span><b>Need a hand?</b> — reopen this guide at any time</span><span><b>API status</b> — shows whether the backend is connected</span></div></div>
          <footer className="guide-footer"><span>Answers are grounded in the resources available in this workspace.</span><button type="button" className="primary-button" onClick={() => setGuideOpen(false)}>Got it</button></footer>
        </section></div>}
    </div>
  );
}

export default App;
