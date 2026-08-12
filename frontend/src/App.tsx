import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { UploadCloud, FileText, Trash2, Paperclip, Copy } from 'lucide-react'

type Message = { id: string; role: 'user'|'assistant'; text: string; sources?: any[] }
type DocumentInfo = { document_id: string; filename: string; size: number; pages: number; status: string }

export default function App(){
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const chatRef = useRef<HTMLDivElement | null>(null)

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(true)
  }

  const handleDragLeave = () => {
    setDragActive(false)
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragActive(false)
    const files = Array.from(event.dataTransfer.files || [])
    if (files.length) {
      onFileSelected(undefined, files)
    }
  }

  useEffect(()=>{
    if(chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight
  },[messages])

  const onFileSelected = async (f?: File, files?: File[]) => {
    const selectedFiles = files?.length ? files : f ? [f] : Array.from(fileInputRef.current?.files ?? [])
    if (selectedFiles.length === 0) return

    const invalid = selectedFiles.find(file => !file.name.toLowerCase().endsWith('.pdf') || file.size > 25*1024*1024)
    if (invalid) {
      alert('Only PDFs under 25 MB are allowed.')
      return
    }

    const form = new FormData()
    selectedFiles.forEach(file => form.append('files', file))
    if (sessionId) {
      form.append('session_id', sessionId)
    }

    try{
      setLoading(true)
      const res = await axios.post('http://localhost:8000/api/upload', form)
      const uploadedDocuments = res.data?.documents ?? []
      if (uploadedDocuments.length === 0) {
        alert('Upload succeeded but no documents were returned.')
        return
      }
      const newSessionId = res.data.session_id
      setSessionId(newSessionId)
      setDocuments(prev => {
        const merged = [...prev, ...uploadedDocuments]
        const uniqueDocs = Array.from(new Map(merged.map(doc => [doc.document_id, doc])).values())
        return uniqueDocs
      })
      setSelectedDocumentId(prevId => prevId ?? uploadedDocuments[0]?.document_id ?? null)
      setMessages([])
    }catch(err:any){
      alert(err?.response?.data?.detail || 'Upload failed')
    }finally{setLoading(false)}
  }

  const sendQuestion = async () =>{
    if(!sessionId) return alert('Upload a document first')
    if(!selectedDocumentId) return alert('Select a document first')
    if(!question.trim()) return
    const id = String(Date.now())
    setMessages(prev=>[...prev, {id, role:'user', text:question}])
    setLoading(true)
    try{
      const res = await axios.post('http://localhost:8000/api/ask', {session_id: sessionId, question, document_id: selectedDocumentId})
      const ans = res.data.answer
      const sources = res.data.sources
      setMessages(prev=>[...prev, {id: id+'-ans', role:'assistant', text: ans, sources}])
      setQuestion('')
    }catch(err:any){
      alert(err?.response?.data?.detail || 'Error asking question')
    }finally{setLoading(false)}
  }

  const clearAll = async () =>{
    if(!sessionId) return
    await axios.post('http://localhost:8000/api/clear', new URLSearchParams({session_id: sessionId}))
    setSessionId(null)
    setDocuments([])
    setSelectedDocumentId(null)
    setMessages([])
  }

  const selectedDocument = documents.find(doc => doc.document_id === selectedDocumentId)

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) =>{
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault(); sendQuestion()
    }
  }

  return (
    <div className="min-h-screen flex bg-slate-50">
      <aside className="w-80 p-6 bg-white shadow-xl border border-slate-200">
        <div className="flex items-center gap-3 mb-5">
          <div className="rounded-2xl bg-brand-500 p-3 text-white shadow-sm"><UploadCloud size={20} /></div>
          <div>
            <h3 className="text-xl font-semibold">Document QA</h3>
            <p className="text-sm text-slate-500">Upload multiple PDFs and ask questions per document.</p>
          </div>
        </div>

        <div className="mb-5">
          <label className="block text-sm font-semibold text-slate-700">Upload PDFs</label>
          <div
            className={`mt-3 border-2 rounded-3xl p-5 text-center transition duration-200 ${dragActive ? 'border-brand-500 bg-brand-50' : 'border-slate-200 bg-white'}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center gap-3 text-slate-500">
              <FileText size={26} />
              <div className="text-sm font-medium">Drag & drop PDFs here</div>
              <div className="text-xs">Upload one or more PDF files (max 25 MB each).</div>
            </div>
            <div className="mt-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="application/pdf"
                multiple
                className="hidden"
                onChange={(e)=>onFileSelected(undefined, Array.from(e.target.files ?? []))}
              />
              <button
                type="button"
                className="mt-2 inline-flex items-center justify-center px-4 py-2 rounded-full bg-brand-500 text-white shadow-sm hover:bg-brand-600"
                onClick={(e)=>{ e.stopPropagation(); fileInputRef.current?.click() }}
              >
                Browse files
              </button>
            </div>
          </div>
        </div>

        {documents.length > 0 ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-700">Uploaded PDFs</p>
                <p className="text-xs text-slate-500">Select a document before asking.</p>
              </div>
              <button
                type="button"
                className="text-sm text-red-600 hover:text-red-700"
                onClick={clearAll}
              >
                Clear all
              </button>
            </div>
            <div className="space-y-3">
              {documents.map(doc => (
                <button
                  key={doc.document_id}
                  type="button"
                  className={`w-full text-left p-4 rounded-3xl border shadow-sm transition ${selectedDocumentId === doc.document_id ? 'border-brand-500 bg-brand-50 shadow-md' : 'border-slate-200 bg-white hover:border-slate-300'}`}
                  onClick={() => setSelectedDocumentId(doc.document_id)}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-slate-800">{doc.filename}</p>
                      <p className="mt-1 text-xs text-slate-500">{Math.round(doc.size/1024)} KB • {doc.pages} pages</p>
                    </div>
                    <div className={`flex items-center justify-center h-8 w-8 rounded-full border ${selectedDocumentId === doc.document_id ? 'border-brand-500 bg-brand-500 text-white' : 'border-slate-300 bg-white text-slate-400'}`}>
                      {selectedDocumentId === doc.document_id ? '✓' : '○'}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-500">Upload documents to select one and start asking questions.</div>
        )}
      </aside>

      <main className="flex-1 p-6">
        <header className="mb-6">
          <h1 className="text-3xl font-semibold text-slate-900">Ask questions about your PDF</h1>
          <p className="mt-2 text-sm text-slate-500">Choose a document and get precise answers from that file only.</p>
        </header>

        <div className="h-[60vh] overflow-auto p-4 bg-white rounded-2xl shadow-inner" ref={chatRef}>
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <Paperclip size={48} />
              <div className="mt-3">Upload a document to start asking questions.</div>
            </div>
          )}

          {messages.map(m=> (
            <div key={m.id} className={`mb-4 max-w-3xl ${m.role==='user' ? 'ml-auto text-right' : ''}`}>
              <div className="inline-block bg-gray-100 p-4 rounded-2xl shadow-sm">
                <div className="whitespace-pre-wrap">{m.text}</div>
              </div>
              {m.sources && (
                <div className="mt-2 text-xs text-gray-500">
                  <div className="font-semibold">Sources:</div>
                  {m.sources.map((s:any, i:number)=> (
                    <div key={i}>Page {s.page} — {s.text}</div>
                  ))}
                </div>
              )}
            </div>
          ))}

          {loading && <div className="text-sm text-gray-500">Processing...</div>}
        </div>

        <div className="mt-4 bg-white p-4 rounded-2xl shadow-md">
          {selectedDocument ? (
            <div className="mb-4 rounded-3xl bg-slate-50 border border-slate-200 p-4 text-sm text-slate-700">
              <div className="font-semibold">Selected document</div>
              <div className="mt-1 text-slate-500">{selectedDocument.filename} • {Math.round(selectedDocument.size/1024)} KB • {selectedDocument.pages} pages</div>
            </div>
          ) : (
            <div className="mb-4 rounded-3xl bg-rose-50 border border-rose-200 p-4 text-sm text-rose-700">
              Select a document from the left panel before asking a question.
            </div>
          )}
          <textarea value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={onKeyDown} placeholder="Ask a question..." className="w-full p-3 rounded-md border" rows={3} />
          <div className="flex items-center justify-between mt-2">
            <div className="text-xs text-gray-500">Shift+Enter for newline • Enter to send</div>
            <div className="flex gap-2">
              <button className="px-4 py-2 bg-gray-100 rounded-md" onClick={()=>{setQuestion('')}}>Clear</button>
              <button className="px-4 py-2 bg-brand-500 text-white rounded-md" onClick={sendQuestion} disabled={loading || !selectedDocumentId}>{loading? 'Loading...':'Send'}</button>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
