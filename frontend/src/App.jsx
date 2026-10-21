import { useEffect, useState } from "react";
import "./App.css";

const API = (
  import.meta.env.BACKEND_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const examples = [
  "Show the latest outlet BOD, COD and TSS results for GRN-STP-011",
  "Show the latest outlet BOD, COD and TSS results for GRN-STP-007",
  "Total STP in Panipat?"
];

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

  async function loadDocuments() {
    setDocumentsLoading(true);
    setDocumentsError("");
    try {
      const response = await fetch(`${API}/api/documents`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Could not load documents."
        );
      }
      setDocuments(data.documents || []);
    } catch (err) {
      setDocumentsError(err.message || "Could not load documents.");
    } finally {
      setDocumentsLoading(false);
    }
  }

  async function deleteDocument(doc) {
    if (!window.confirm(`Delete "${doc.original_filename}" and its indexed records?`)) {
      return;
    }
    setDeletingFile(doc.stored_filename);
    setDocumentsError("");
    try {
      const response = await fetch(
        `${API}/api/documents/${encodeURIComponent(doc.stored_filename)}`,
        { method: "DELETE" }
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string" ? data.detail : "Could not delete document."
        );
      }
      setDocuments((current) =>
        current.filter((item) => item.stored_filename !== doc.stored_filename)
      );
      setUploadMessage(`Deleted ${doc.original_filename} successfully.`);
    } catch (err) {
      setDocumentsError(err.message || "Could not delete document.");
    } finally {
      setDeletingFile("");
    }
  }

  async function reindexDocument(doc) {
    setDeletingFile(doc.stored_filename);
    setDocumentsError("");
    setUploadMessage("");
    try {
      const response = await fetch(
        `${API}/api/documents/${encodeURIComponent(doc.stored_filename)}/reindex`,
        { method: "POST" }
      );
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : data.detail?.message || "Could not re-index document."
        );
      }
      setUploadMessage(`${doc.original_filename} re-indexed (${data.indexed_records} records).`);
      await loadDocuments();
    } catch (err) {
      setDocumentsError(err.message || "Could not re-index document.");
    } finally {
      setDeletingFile("");
    }
  }

  function fileLabel(name) {
    const match = /\.([A-Za-z0-9]+)$/.exec(name || "");
    return match ? match[1].toUpperCase().slice(0, 4) : "FILE";
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then((r) => {
        if (!r.ok) throw new Error("Backend unavailable");
        return r.json();
      })
      .then(() => setBackendStatus("Connected"))
      .catch(() => setBackendStatus("Disconnected"));
  }, []);

  async function uploadDocument(e) {
    e.preventDefault();
    if (!selectedFile || uploading) return;

    setUploading(true);
    setUploadMessage("");
    setUploadError("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API}/api/documents/upload`, {
        method: "POST",
        body: formData
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const detail = typeof data.detail === "string"
          ? data.detail
          : data.detail?.message || `Upload failed (${response.status})`;
        throw new Error(detail);
      }

      if (!data.indexed_records || data.indexed_records <= 0) {
        throw new Error(
          `${data.original_filename} was saved but no content could be indexed.`
        );
      }

      setUploadMessage(
        `${data.original_filename} uploaded and indexed successfully (${data.indexed_records} records).`
      );
      await loadDocuments();
      setSelectedFile(null);
      e.target.reset();
    } catch (err) {
      setUploadError(err.message || "Could not upload the file.");
      loadDocuments();
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion(e) {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.trim() })
      });

      if (!response.ok) {
        throw new Error(`Request failed (${response.status})`);
      }

      setResult(await response.json());
    } catch (err) {
      setError(
        `${err.message}. Check that the backend URL is configured and reachable.`
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">W</div>
          <div>
            <h2>Wastewater RAG</h2>
            <p>Wastewater Intelligence Platform</p>
          </div>
        </div>
        <span className={`status ${backendStatus === "Connected" ? "status-connected" : "status-disconnected"}`}>
          API: {backendStatus}
        </span>
      </header>

      <main className="main-content">
        <section className="hero">
          <span className="eyebrow">RETRIEVAL-AUGMENTED GENERATION</span>
          <h1>Wastewater Treatment Intelligence</h1>
          <p>
            Ask questions about wastewater treatment plants, water-quality
            samples and technical reports. Explore answers with source references.
          </p>
        </section>

        <section className="ask-panel upload-panel">
          <h2>Upload Documents</h2>
          <p className="muted">
            Add wastewater reports and datasets in PDF, Excel, CSV, DOCX or TXT format.
            Maximum file size: 25 MB.
          </p>
          <form onSubmit={uploadDocument}>
            <label htmlFor="document-file">Choose a document</label>
            <input
              id="document-file"
              type="file"
              accept=".pdf,.xlsx,.csv,.docx,.txt"
              onChange={(e) => {
                setSelectedFile(e.target.files?.[0] || null);
                setUploadMessage("");
                setUploadError("");
              }}
              required
            />
            <button type="submit" disabled={!selectedFile || uploading}>
              {uploading ? "Uploading..." : "Upload Document"}
            </button>
          </form>
          {uploadMessage && <div className="success-panel" role="status">{uploadMessage}</div>}
          {uploadError && <div className="error-panel" role="alert">{uploadError}</div>}
        </section>

        <section className="ask-panel documents-panel">
          <div className="documents-heading">
            <div>
              <h2>Uploaded Documents</h2>
              <p className="muted">
                View uploaded files and remove files and their indexed records.
              </p>
            </div>
            <button
              type="button"
              className="refresh-button"
              onClick={loadDocuments}
              disabled={documentsLoading}
            >
              {documentsLoading ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          <div className="documents-count">
            {documents.length} document{documents.length === 1 ? "" : "s"}
          </div>

          {documentsError && (
            <div className="error-panel" role="alert">{documentsError}</div>
          )}

          {documentsLoading && documents.length === 0 && (
            <p className="muted">Loading documents...</p>
          )}

          {!documentsLoading && documents.length === 0 && !documentsError && (
            <p className="empty-documents">
              No uploaded documents found. Upload a file above to get started.
            </p>
          )}

          {documents.length > 0 && (
            <div className="document-list">
              {documents.map((doc) => (
                <div className="document-item" key={doc.stored_filename}>
                  <div className="document-icon" aria-hidden="true">{fileLabel(doc.original_filename)}</div>
                  <div className="document-info">
                    <strong title={doc.original_filename}>
                      {doc.original_filename}
                    </strong>
                    <span>
                      {formatFileSize(doc.size_bytes)} · {doc.indexed_records} indexed records
                    </span>
                    <span className={
                      doc.status === "indexed"
                        ? "document-status indexed"
                        : "document-status not-indexed"
                    }>
                      {doc.status === "indexed" ? "Indexed and searchable" : "Not currently indexed"}
                    </span>
                  </div>
                  {doc.status !== "indexed" && (
                    <button
                      type="button"
                      className="refresh-button"
                      onClick={() => reindexDocument(doc)}
                      disabled={Boolean(deletingFile)}
                      aria-label={`Re-index ${doc.original_filename}`}
                    >
                      {deletingFile === doc.stored_filename ? "Working..." : "Re-index"}
                    </button>
                  )}
                  <button
                    type="button"
                    className="delete-button"
                    onClick={() => deleteDocument(doc)}
                    disabled={Boolean(deletingFile)}
                    aria-label={`Delete ${doc.original_filename}`}
                  >
                    {deletingFile === doc.stored_filename ? "Deleting..." : "Delete"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="ask-panel">
          <h2>Ask Wastewater RAG</h2>
          <p className="muted">
            Ask about an STP, sample result, capacity or treatment report, or about anything in your uploaded files.
          </p>

          <form onSubmit={askQuestion}>
            <label htmlFor="question">Your question</label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. Show the latest outlet BOD, COD and TSS results for GRN-STP-011"
              rows={3}
              required
            />
            <button type="submit" disabled={loading || !question.trim()}>
              {loading ? "Searching documents..." : "Ask Question"}
            </button>
          </form>

          <div className="examples">
            <p>Try an example</p>
            {examples.map((example) => (
              <button
                type="button"
                className="example-button"
                key={example}
                onClick={() => setQuestion(example)}
              >
                {example}
              </button>
            ))}
          </div>
        </section>

        {loading && <div className="message">Searching your indexed documents. Please wait...</div>}
        {error && <div className="error-panel">{error}</div>}

        {result && (
          <section className="answer-panel">
            <span className="eyebrow">RESPONSE</span>
            <h2>Answer</h2>
            <p className="answer-text">{result.answer || "No answer was returned."}</p>

            {result.summary && (
              <div className="summary">
                <strong>Records found:</strong>{" "}
                {result.summary.records_count ?? "Not specified"}
                {result.summary.sample_results_found === false &&
                  " · No dated sample results found"}
              </div>
            )}

            {Array.isArray(result.records) && result.records.some(
              (record) => record.stp_id || record.stp_name || record.sample_section
            ) && (
              <div className="table-wrap">
                <h3>Retrieved records</h3>
                <table>
                  <thead>
                    <tr>
                      <th>STP ID</th>
                      <th>Sample type</th>
                      <th>Date</th>
                      <th>BOD</th>
                      <th>COD</th>
                      <th>TSS</th>
                      <th>pH</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.records.map((record, index) => (
                      <tr key={`${record.stp_id || "record"}-${index}`}>
                        <td>{record.stp_id || record.stp_name || "—"}</td>
                        <td>{record.sample_section || "—"}</td>
                        <td>{record.sample_date || "—"}</td>
                        <td>{record.bod ?? "—"}</td>
                        <td>{record.cod ?? "—"}</td>
                        <td>{record.tss ?? "—"}</td>
                        <td>{record.ph ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {Array.isArray(result.sources) && result.sources.length > 0 && (
              <div className="sources">
                <h3>Sources</h3>
                {result.sources.map((source, index) => (
                  <details key={source.source_id ?? index}>
                    <summary>
                      {source.document || source.source || `Source ${index + 1}`}
                      {source.page && source.page !== "N/A" ? ` · Page ${source.page}` : ""}
                      {source.table_number && source.table_number !== "N/A" ? ` · Table ${source.table_number}` : ""}
                    </summary>
                    <pre>{source.content || source.text || "Source content not provided."}</pre>
                  </details>
                ))}
              </div>
            )}
          </section>
        )}

        <footer>
          Wastewater RAG · Research and information retrieval prototype
        </footer>
      </main>
    </div>
  );
}

export default App;
