import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { UploadCloud, FileText, Trash2, Paperclip, Copy } from 'lucide-react'

type Message = { id: string; role: 'user'|'assistant'; text: string; sources?: any[] }

export default function App(){
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [fileMeta, setFileMeta] = useState<any>(null)
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
    const file = event.dataTransfer.files?.[0]
    if (file) {
      onFileSelected(file)
    }
  }

  useEffect(()=>{
    if(chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight
  },[messages])

  const onFileSelected = async (f?: File) => {
    const file = f || (fileInputRef.current?.files?.[0] ?? null)
    if(!file) return
    if(file.size > 25*1024*1024){
      alert('File exceeds maximum upload size of 25 MB.')
      return
    }
    if(!file.name.toLowerCase().endsWith('.pdf')){
      alert('Unsupported file type. Only PDFs allowed.')
      return
    }
    const form = new FormData()
    form.append('file', file)
    try{
      setLoading(true)
      const res = await axios.post('http://localhost:8000/api/upload', form, { headers: {'Content-Type':'multipart/form-data'}})
      setSessionId(res.data.session_id)
      setFileMeta({filename: res.data.filename, size: res.data.size, pages: res.data.pages})
      setMessages([])
    }catch(err:any){
      alert(err?.response?.data?.detail || 'Upload failed')
    }finally{setLoading(false)}
  }

  const sendQuestion = async () =>{
    if(!sessionId) return alert('Upload a document first')
    if(!question.trim()) return
    const id = String(Date.now())
    setMessages(prev=>[...prev, {id, role:'user', text:question}])
    setLoading(true)
    try{
      const res = await axios.post('http://localhost:8000/api/ask', {session_id: sessionId, question})
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
    setFileMeta(null)
    setMessages([])
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) =>{
    if(e.key === 'Enter' && !e.shiftKey){
      e.preventDefault(); sendQuestion()
    }
  }

  return (
    <div className="min-h-screen flex">
      <aside className="w-80 p-6 bg-white shadow-lg">
        <div className="flex items-center gap-2 mb-4">
          <UploadCloud />
          <h3 className="text-lg font-semibold">RAG PDF QA</h3>
        </div>
        <div className="mb-4">
          <label className="block text-sm text-gray-600">Upload</label>
          <div
            className={`mt-2 border-dashed border-2 rounded-2xl text-center p-4 ${dragActive ? 'border-brand-500 bg-brand-50' : 'border-gray-200 bg-white'} cursor-pointer transition-colors duration-200`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="flex items-center justify-center gap-2 text-gray-500">
              <FileText />
              <div>
                <div className="font-medium">Drag & drop your document</div>
                <div className="text-xs">Maximum 100 pages • PDF</div>
              </div>
            </div>
            <div className="mt-3">
              <input ref={fileInputRef} type="file" accept="application/pdf" className="hidden" onChange={(e)=>onFileSelected(e.target.files?.[0] ?? undefined)} />
              <button className="mt-2 px-4 py-2 bg-brand-500 text-white rounded-md cursor-pointer" onClick={(e)=>{ e.stopPropagation(); fileInputRef.current?.click() }}>Browse</button>
            </div>
          </div>
        </div>

        {fileMeta ? (
          <div className="mt-4 bg-gray-50 p-3 rounded-lg">
            <div className="flex justify-between items-start">
              <div>
                <div className="text-sm font-medium">{fileMeta.filename}</div>
                <div className="text-xs text-gray-500">{Math.round(fileMeta.size/1024)} KB • {fileMeta.pages} pages</div>
              </div>
              <button className="text-red-500" onClick={clearAll}><Trash2 /></button>
            </div>
          </div>
        ) : (
          <div className="mt-6 text-sm text-gray-500">Upload a document to start asking questions.</div>
        )}

      </aside>

      <main className="flex-1 p-6">
        <header className="mb-4">
          <h1 className="text-2xl font-semibold">Ask questions about your PDF</h1>
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
          <textarea value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={onKeyDown} placeholder="Ask a question..." className="w-full p-3 rounded-md border" rows={3} />
          <div className="flex items-center justify-between mt-2">
            <div className="text-xs text-gray-500">Shift+Enter for newline • Enter to send</div>
            <div className="flex gap-2">
              <button className="px-4 py-2 bg-gray-100 rounded-md" onClick={()=>{setQuestion('')}}>Clear</button>
              <button className="px-4 py-2 bg-brand-500 text-white rounded-md" onClick={sendQuestion} disabled={loading}>{loading? 'Loading...':'Send'}</button>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
