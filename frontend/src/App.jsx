import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import {
  Search, Briefcase, MapPin, GitBranch, Shield, TrendingUp,
  Star, BarChart3, LayoutGrid, GitCompare, User, ChevronRight,
  CheckCircle2, AlertTriangle, Award, Clock, Building2, GraduationCap,
  Upload, FileText, Trash2, Play, Loader2, AlertCircle, ChevronDown
} from 'lucide-react'
import './index.css'

const API_BASE = 'http://localhost:8000'

/* ──────────────────────────────────────────────────────────────
   Utility helpers
   ────────────────────────────────────────────────────────────── */

function scoreColor(score) {
  if (score >= 0.85) return 'var(--green)'
  if (score >= 0.70) return 'var(--amber)'
  return 'var(--red)'
}

function proficiencyClass(proficiency) {
  switch ((proficiency || '').toLowerCase()) {
    case 'expert':       return 'expert'
    case 'advanced':     return 'advanced'
    case 'intermediate': return 'intermediate'
    default:             return ''
  }
}

function formatDate(dateStr) {
  if (!dateStr) return 'Present'
  const d = new Date(dateStr)
  return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

/* Parse reasoning into structured segments */
function parseReasoning(text) {
  return text.split(';').map(s => s.trim()).filter(Boolean)
}

/* ──────────────────────────────────────────────────────────────
   Score breakdown for each candidate (derived from reasoning)
   ────────────────────────────────────────────────────────────── */

function buildScoreBars(c) {
  const sig    = c.profile_data?.redrob_signals || {}
  const prof   = c.profile_data?.profile || {}
  const skills = c.profile_data?.skills || []
  const career = c.profile_data?.career_history || []

  const yoe    = parseFloat(prof.years_of_experience || 0)
  const github = sig.github_activity_score >= 0 ? sig.github_activity_score : null
  const response = (sig.recruiter_response_rate || 0) * 100

  // Semantic score from reasoning
  const semMatch = (() => {
    const m = c.reasoning.match(/semantic match ([\d.]+)/)
    return m ? parseFloat(m[1]) * 100 : 0
  })()

  const expertSkills = skills.filter(s => s.proficiency === 'expert').length
  const advancedSkills = skills.filter(s => s.proficiency === 'advanced').length
  const skillScore = Math.min(100, (expertSkills * 15 + advancedSkills * 8))

  const careerScore = Math.min(100, career.reduce((sum, j) => {
    const ml = ['machine learning','nlp','retrieval','ranking','recommendation','embedding','llm']
    const title = (j.title || '').toLowerCase()
    return sum + (ml.some(k => title.includes(k)) ? 30 : 10)
  }, 0))

  const expScore = Math.min(100, yoe >= 7 ? 100 : yoe >= 5 ? 85 : yoe >= 3 ? 65 : 40)

  return [
    { label: 'Semantic Match',    value: semMatch,      color: 'var(--accent)' },
    { label: 'Skill Strength',    value: skillScore,    color: 'var(--purple)' },
    { label: 'Career Relevance',  value: careerScore,   color: 'var(--blue)' },
    { label: 'Experience Fit',    value: expScore,      color: 'var(--green)' },
    { label: 'GitHub Activity',   value: github ?? 0,   color: 'var(--amber)', hide: github === null },
    { label: 'Recruiter Response',value: response,      color: 'var(--green)' },
  ].filter(b => !b.hide)
}

/* ──────────────────────────────────────────────────────────────
   Skill heatmap: count JD-relevant skills across all candidates
   ────────────────────────────────────────────────────────────── */

const JD_SKILL_KEYWORDS = [
  'python','llm','rag','embedding','vector','pytorch','tensorflow','transformers',
  'mlops','docker','kubernetes','faiss','nlp','fine-tun','lora','langchain',
  'ranking','retrieval','semantic','recommendation','huggingface','xgboost','lightgbm',
  'weaviate','pinecone','milvus','qdrant','spark','kafka','redis','fastapi',
  'learning to rank','deep learning','reinforcement learning',
]

function buildHeatmap(candidates) {
  const freq = {}
  candidates.forEach(c => {
    const skills = c.profile_data?.skills || []
    const seen = new Set()
    skills.forEach(s => {
      const n = (s.name || '').toLowerCase()
      JD_SKILL_KEYWORDS.forEach(kw => {
        if (n.includes(kw) && !seen.has(kw)) {
          seen.add(kw)
          freq[kw] = (freq[kw] || 0) + 1
        }
      })
    })
  })
  return Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 30)
}

/* ──────────────────────────────────────────────────────────────
   PROFILE VIEW
   ────────────────────────────────────────────────────────────── */

function ProfileView({ candidate }) {
  if (!candidate) {
    return (
      <div className="empty-state">
        <User size={40} />
        <span>Select a candidate from the list</span>
        <p>Click any row in the sidebar to view their full profile</p>
      </div>
    )
  }

  const { rank, score, reasoning, profile_data } = candidate
  const prof   = profile_data.profile   || {}
  const career = profile_data.career_history || []
  const skills = profile_data.skills    || []
  const edu    = profile_data.education || []
  const sig    = profile_data.redrob_signals || {}

  const bars   = buildScoreBars(candidate)
  const notes  = parseReasoning(reasoning)

  const github     = sig.github_activity_score >= 0 ? sig.github_activity_score : null
  const isOpenToWork = sig.open_to_work_flag
  const responseRate = ((sig.recruiter_response_rate || 0) * 100).toFixed(0)

  return (
    <div className="animate-in" key={profile_data.candidate_id}>
      {/* Header */}
      <div className="profile-header">
        <div className="profile-meta">
          <div className="profile-rank">Rank #{rank} · {profile_data.candidate_id}</div>
          <div className="profile-title">{prof.current_title}</div>
          <div className="profile-badges">
            <span className="badge badge-neutral"><Building2 size={12} /> {prof.current_company}</span>
            <span className="badge badge-neutral"><MapPin size={12} /> {prof.location}</span>
            <span className="badge badge-neutral"><Briefcase size={12} /> {prof.years_of_experience} yrs</span>
            {isOpenToWork && <span className="badge badge-green"><CheckCircle2 size={12} /> Open to Work</span>}
            {github !== null && (
              <span className="badge badge-purple"><GitBranch size={12} /> GitHub {github}/100</span>
            )}
          </div>
        </div>
        <div className="score-display">
          <div className="score-num" style={{ color: scoreColor(score) }}>
            {(score * 100).toFixed(1)}
          </div>
          <div className="score-label">Match Score</div>
        </div>
      </div>

      {/* AI Reasoning */}
      <div className="section">
        <div className="section-title"><TrendingUp size={14} /> AI Recruiter Analysis</div>
        <div className="reasoning-card">
          {notes.map((note, i) => (
            <div key={i} style={{ marginBottom: i < notes.length - 1 ? '6px' : 0 }}>
              <span style={{ color: 'var(--green)', marginRight: '8px' }}>·</span>
              {note}
            </div>
          ))}
        </div>
      </div>

      {/* Quick Metrics */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value" style={{ color: scoreColor(score) }}>
            {(score * 100).toFixed(1)}%
          </div>
          <div className="metric-label">Overall Match</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: 'var(--blue)' }}>{prof.years_of_experience}</div>
          <div className="metric-label">Years Experience</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: 'var(--purple)' }}>{github ?? 'N/A'}</div>
          <div className="metric-label">GitHub Score</div>
        </div>
        <div className="metric-card">
          <div className="metric-value" style={{ color: 'var(--amber)' }}>{responseRate}%</div>
          <div className="metric-label">Response Rate</div>
        </div>
      </div>

      {/* Score Breakdown */}
      <div className="section">
        <div className="section-title"><BarChart3 size={14} /> Score Breakdown</div>
        <div className="score-bars">
          {bars.map(bar => (
            <div className="score-bar-row" key={bar.label}>
              <div className="score-bar-label">{bar.label}</div>
              <div className="score-bar-track">
                <div
                  className="score-bar-fill"
                  style={{ width: `${bar.value}%`, background: bar.color }}
                />
              </div>
              <div className="score-bar-num">{bar.value.toFixed(0)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Education */}
      {edu.length > 0 && (
        <div className="section">
          <div className="section-title"><GraduationCap size={14} /> Education</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {edu.map((e, i) => (
              <div key={i} className="job-card" style={{ flex: '1 1 240px' }}>
                <div className="job-title-text">{e.institution}</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-2)', marginTop: '3px' }}>
                  {e.degree} · {e.field_of_study}
                </div>
                {e.tier && (
                  <span className={`badge ${e.tier === 'tier_1' ? 'badge-green' : e.tier === 'tier_2' ? 'badge-blue' : 'badge-neutral'}`}
                    style={{ marginTop: '8px' }}>
                    {e.tier.replace('_', ' ')}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Skills */}
      <div className="section">
        <div className="section-title"><Star size={14} /> Skills ({skills.length})</div>
        <div className="skills-grid">
          {skills.map((s, i) => (
            <span key={i} className={`skill-chip ${proficiencyClass(s.proficiency)}`}>
              {s.name}
              {s.endorsements > 0 && (
                <span className="skill-endorsements">+{s.endorsements}</span>
              )}
            </span>
          ))}
        </div>
      </div>

      {/* Career Timeline */}
      <div className="section">
        <div className="section-title"><Clock size={14} /> Career History</div>
        <div className="timeline">
          {career.map((job, i) => (
            <div key={i} className={`timeline-item ${job.is_current ? 'current' : ''}`}>
              <div className="timeline-dot" />
              <div className="job-card">
                <div className="job-header">
                  <div>
                    <div className="job-title-text">{job.title}</div>
                    <div className="job-company-text">{job.company}</div>
                  </div>
                  <div className="job-date">
                    {formatDate(job.start_date)} – {formatDate(job.end_date)}
                    <br />
                    <span style={{ color: 'var(--text-3)' }}>{job.duration_months}mo</span>
                  </div>
                </div>
                {job.description && (
                  <div className="job-desc">{job.description}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────
   HEATMAP VIEW
   ────────────────────────────────────────────────────────────── */

function HeatmapView({ candidates }) {
  const data = useMemo(() => buildHeatmap(candidates), [candidates])
  const maxVal = data[0]?.[1] || 1

  return (
    <div className="animate-in">
      <div style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '6px' }}>Skill Coverage Heatmap</h2>
        <p style={{ color: 'var(--text-2)', fontSize: '0.9rem' }}>
          How frequently each JD-relevant skill appears across the Top {candidates.length} candidates
        </p>
      </div>
      <div className="heatmap-grid">
        {data.map(([skill, count]) => (
          <div className="heat-cell" key={skill}>
            <div className="heat-skill-name">{skill}</div>
            <div className="heat-bar-track">
              <div
                className="heat-bar-fill"
                style={{
                  width: `${(count / maxVal) * 100}%`,
                  background: count > maxVal * 0.7 ? 'var(--green)' : count > maxVal * 0.4 ? 'var(--accent)' : 'var(--blue)'
                }}
              />
            </div>
            <div className="heat-count">{count} / {candidates.length} candidates</div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────
   COMPARE VIEW
   ────────────────────────────────────────────────────────────── */

function CompareView({ candidates }) {
  if (candidates.length < 2) {
    return (
      <div className="compare-hint animate-in">
        <GitCompare size={48} style={{ opacity: 0.2 }} />
        <h3 style={{ fontWeight: 600 }}>Select 2 candidates to compare</h3>
        <p style={{ color: 'var(--text-3)', fontSize: '0.85rem', maxWidth: '300px' }}>
          Hold <kbd style={{ background: 'var(--surface-3)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.78rem' }}>Ctrl</kbd> and click two candidates in the sidebar to compare them side by side.
        </p>
      </div>
    )
  }

  const [a, b] = candidates

  function StatRow({ label, aVal, bVal, higherIsBetter = true }) {
    const aNum = parseFloat(aVal) || 0
    const bNum = parseFloat(bVal) || 0
    const aBetter = higherIsBetter ? aNum >= bNum : aNum <= bNum
    const bBetter = higherIsBetter ? bNum >= aNum : bNum <= aNum
    return (
      <>
        <div className="compare-stat-row">
          <span className="compare-stat-label">{label}</span>
          <span className={`compare-stat-value ${aBetter ? 'better' : 'worse'}`}>{aVal}</span>
        </div>
        <div className="compare-stat-row" style={{ borderColor: 'transparent' }}>
          <span className="compare-stat-label" />
          <span className={`compare-stat-value ${bBetter ? 'better' : 'worse'}`}>{bVal}</span>
        </div>
      </>
    )
  }

  function ComparePanel({ c }) {
    const prof   = c.profile_data?.profile || {}
    const skills = c.profile_data?.skills || []
    const career = c.profile_data?.career_history || []
    const sig    = c.profile_data?.redrob_signals || {}
    const edu    = c.profile_data?.education || []
    const topSkills = skills.slice(0, 8).map(s => s.name).join(', ')

    return (
      <div>
        <div className="compare-header">
          <div style={{ color: 'var(--text-2)', fontSize: '0.78rem', marginBottom: '4px' }}>Rank #{c.rank}</div>
          <div style={{ fontWeight: 700, fontSize: '1.1rem', lineHeight: 1.2 }}>{prof.current_title}</div>
          <div style={{ color: 'var(--text-2)', fontSize: '0.82rem', marginTop: '4px' }}>
            {prof.current_company} · {prof.location}
          </div>
        </div>

        <div className="compare-section-title">Core Metrics</div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Match Score</span>
          <span className="compare-stat-value" style={{ color: scoreColor(c.score) }}>
            {(c.score * 100).toFixed(1)}%
          </span>
        </div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Experience</span>
          <span className="compare-stat-value">{prof.years_of_experience} yrs</span>
        </div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">GitHub Score</span>
          <span className="compare-stat-value">
            {sig.github_activity_score >= 0 ? sig.github_activity_score : 'N/A'}
          </span>
        </div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Response Rate</span>
          <span className="compare-stat-value">
            {((sig.recruiter_response_rate || 0) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Open to Work</span>
          <span className={`compare-stat-value ${sig.open_to_work_flag ? 'better' : 'worse'}`}>
            {sig.open_to_work_flag ? 'Yes' : 'No'}
          </span>
        </div>

        <div className="compare-section-title">Profile</div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Company</span>
          <span className="compare-stat-value" style={{ color: 'var(--accent)' }}>{prof.current_company}</span>
        </div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Industry</span>
          <span className="compare-stat-value">{prof.current_industry}</span>
        </div>
        <div className="compare-stat-row">
          <span className="compare-stat-label">Education</span>
          <span className="compare-stat-value">{edu[0]?.tier?.replace('_', ' ') || 'N/A'}</span>
        </div>

        <div className="compare-section-title">Top Skills</div>
        <div style={{ fontSize: '0.82rem', color: 'var(--text-2)', lineHeight: 1.6 }}>{topSkills}</div>

        <div className="compare-section-title">Career Snapshot</div>
        {career.slice(0, 3).map((j, i) => (
          <div key={i} style={{ fontSize: '0.82rem', padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
            <div style={{ fontWeight: 600 }}>{j.title}</div>
            <div style={{ color: 'var(--text-2)' }}>{j.company} · {j.duration_months}mo</div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="animate-in">
      <div style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '6px' }}>Candidate Comparison</h2>
        <p style={{ color: 'var(--text-2)', fontSize: '0.9rem' }}>Side-by-side breakdown of two selected candidates</p>
      </div>
      <div className="compare-grid">
        <ComparePanel c={a} />
        <ComparePanel c={b} />
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────
   UPLOAD VIEW
   ────────────────────────────────────────────────────────────── */

function UploadView({ onResultsReady }) {
  const [jdText,    setJdText]    = useState('')
  const [files,     setFiles]     = useState([])
  const [dragging,  setDragging]  = useState(false)
  const [jobId,     setJobId]     = useState(null)
  const [progress,  setProgress]  = useState(0)
  const [status,    setStatus]    = useState('idle') // idle | uploading | running | done | error
  const [error,     setError]     = useState('')
  const [jdSkills,  setJdSkills]  = useState([])
  const fileInputRef = useRef()
  const pollRef      = useRef()

  const addFiles = useCallback((incoming) => {
    const allowed = Array.from(incoming).filter(f =>
      f.name.endsWith('.pdf') || f.name.endsWith('.docx') || f.name.endsWith('.doc')
    )
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name))
      const fresh = allowed.filter(f => !existing.has(f.name))
      return [...prev, ...fresh].slice(0, 50)
    })
  }, [])

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }

  async function handleSubmit() {
    if (!jdText.trim()) { setError('Please paste a job description.'); return }
    if (files.length === 0) { setError('Please upload at least one resume.'); return }
    setError('')
    setStatus('uploading')
    setProgress(5)

    const fd = new FormData()
    fd.append('jd_text', jdText)
    files.forEach(f => fd.append('resumes', f))

    try {
      const res = await fetch(`${API_BASE}/api/rank`, { method: 'POST', body: fd })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      setJobId(data.job_id)
      setStatus('running')
      setProgress(10)

      // Poll for result
      pollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`${API_BASE}/api/status/${data.job_id}`)
          const sd = await sr.json()
          setProgress(10 + sd.progress * 0.9)
          if (sd.jd_skills?.length) setJdSkills(sd.jd_skills)
          if (sd.status === 'done') {
            clearInterval(pollRef.current)
            setStatus('done')
            setProgress(100)
            onResultsReady(sd.results || [])
          }
          if (sd.status === 'error') {
            clearInterval(pollRef.current)
            setStatus('error')
            setError(sd.error || 'Pipeline error')
          }
        } catch (e) {
          clearInterval(pollRef.current)
          setStatus('error')
          setError('Lost connection to server.')
        }
      }, 1200)
    } catch (e) {
      setStatus('error')
      setError(e.message)
    }
  }

  function reset() {
    clearInterval(pollRef.current)
    setStatus('idle')
    setProgress(0)
    setJobId(null)
    setError('')
    setJdSkills([])
  }

  const isRunning = status === 'uploading' || status === 'running'

  return (
    <div className="animate-in" style={{ maxWidth: '800px', margin: '0 auto' }}>
      <div style={{ marginBottom: '28px' }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, marginBottom: '8px' }}>Rank Any Resumes</h2>
        <p style={{ color: 'var(--text-2)', lineHeight: 1.6 }}>
          Paste a job description, upload up to 50 PDF or DOCX resumes, and get an
          AI-powered shortlist in seconds — fraud-detected, semantically matched, and explained.
        </p>
      </div>

      {/* JD Box */}
      <div className="section">
        <div className="section-title"><FileText size={14} /> Job Description</div>
        <textarea
          value={jdText}
          onChange={e => setJdText(e.target.value)}
          disabled={isRunning}
          placeholder="Paste the full job description here. The engine will extract required skills, experience level, and role type automatically…"
          style={{
            width: '100%', minHeight: '160px',
            background: 'var(--surface-2)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: '14px 16px',
            color: 'var(--text)', fontSize: '0.9rem', lineHeight: 1.6,
            fontFamily: 'inherit', resize: 'vertical', outline: 'none',
            transition: 'border-color 0.2s'
          }}
          onFocus={e => e.target.style.borderColor = 'var(--accent)'}
          onBlur={e  => e.target.style.borderColor = 'var(--border)'}
        />
        {jdSkills.length > 0 && (
          <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-3)', alignSelf: 'center' }}>Detected:</span>
            {jdSkills.map(s => (
              <span key={s} className="skill-chip advanced" style={{ fontSize: '0.75rem', padding: '3px 8px' }}>{s}</span>
            ))}
          </div>
        )}
      </div>

      {/* Drop Zone */}
      <div className="section">
        <div className="section-title"><Upload size={14} /> Resumes ({files.length}/50)</div>
        <div
          onDrop={handleDrop}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onClick={() => !isRunning && fileInputRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? 'var(--accent)' : 'var(--border)'}`,
            borderRadius: 'var(--radius)', padding: '32px',
            textAlign: 'center', cursor: isRunning ? 'default' : 'pointer',
            background: dragging ? 'var(--accent-dim)' : 'var(--surface-2)',
            transition: 'all 0.2s', marginBottom: files.length > 0 ? '12px' : 0,
          }}
        >
          <Upload size={28} style={{ opacity: 0.3, margin: '0 auto 10px' }} />
          <div style={{ fontWeight: 600, marginBottom: '4px' }}>Drag & drop resumes here</div>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-2)' }}>PDF or DOCX, up to 50 files</div>
        </div>
        <input
          ref={fileInputRef}
          type="file" multiple accept=".pdf,.docx,.doc"
          style={{ display: 'none' }}
          onChange={e => addFiles(e.target.files)}
        />
        {files.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {files.map((f, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '8px 12px', background: 'var(--surface-2)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                fontSize: '0.83rem'
              }}>
                <FileText size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                <span style={{ color: 'var(--text-3)', fontSize: '0.75rem' }}>{(f.size / 1024).toFixed(0)}KB</span>
                {!isRunning && (
                  <button
                    onClick={e => { e.stopPropagation(); setFiles(prev => prev.filter((_, j) => j !== i)) }}
                    style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-3)', padding: 0 }}
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: 'var(--red-dim)', border: '1px solid rgba(244,63,94,0.3)',
          borderRadius: 'var(--radius)', padding: '12px 16px', marginBottom: '16px',
          display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.85rem', color: 'var(--red)'
        }}>
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {/* Progress */}
      {isRunning && (
        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.82rem', color: 'var(--text-2)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Loader2 size={14} style={{ animation: 'spin 1s linear infinite' }} />
              {status === 'uploading' ? 'Uploading resumes…' : 'Ranking candidates…'}
            </span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div style={{ height: '6px', background: 'var(--surface-3)', borderRadius: '99px', overflow: 'hidden' }}>
            <div style={{ height: '100%', width: `${progress}%`, background: 'var(--accent)', borderRadius: '99px', transition: 'width 0.5s ease' }} />
          </div>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={handleSubmit}
          disabled={isRunning}
          style={{
            flex: 1, padding: '12px 24px', background: 'var(--accent)',
            border: 'none', borderRadius: 'var(--radius)', color: '#fff',
            fontFamily: 'inherit', fontSize: '0.95rem', fontWeight: 600,
            cursor: isRunning ? 'not-allowed' : 'pointer', opacity: isRunning ? 0.6 : 1,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
            transition: 'all 0.2s'
          }}
        >
          {isRunning ? <><Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} /> Running…</> : <><Play size={16} /> Rank These Resumes</>}
        </button>
        {status !== 'idle' && (
          <button
            onClick={reset}
            style={{
              padding: '12px 16px', background: 'var(--surface-2)',
              border: '1px solid var(--border)', borderRadius: 'var(--radius)',
              color: 'var(--text-2)', fontFamily: 'inherit', fontSize: '0.9rem',
              cursor: 'pointer'
            }}
          >
            Reset
          </button>
        )}
      </div>

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────
   APP ROOT
   ────────────────────────────────────────────────────────────── */

export default function App() {
  const [candidates, setCandidates] = useState([])
  const [loading,    setLoading]    = useState(true)
  const [query,      setQuery]      = useState('')
  const [filter,     setFilter]     = useState('all')
  const [selected,   setSelected]   = useState(null)
  const [compareSet, setCompareSet] = useState([])
  const [view,       setView]       = useState('upload')
  const [liveResults, setLiveResults] = useState(null)

  const activeCandidates = liveResults || candidates

  useEffect(() => {
    fetch('/top_candidates.json')
      .then(r => r.json())
      .then(data => {
        setCandidates(data)
        setSelected(data[0] || null)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  function handleResultsReady(results) {
    setLiveResults(results)
    setSelected(results[0] || null)
    setView('profile')
  }

  const filtered = useMemo(() => {
    let list = activeCandidates
    if (query) {
      const q = query.toLowerCase()
      list = list.filter(c => {
        const prof = c.profile_data?.profile || {}
        const skills = (c.profile_data?.skills || []).map(s => s.name.toLowerCase()).join(' ')
        return (
          (prof.current_title || '').toLowerCase().includes(q) ||
          (prof.current_company || '').toLowerCase().includes(q) ||
          (prof.location || '').toLowerCase().includes(q) ||
          skills.includes(q)
        )
      })
    }
    if (filter === 'github') {
      list = list.filter(c => (c.profile_data?.redrob_signals?.github_activity_score || 0) >= 70)
    }
    if (filter === 'open') {
      list = list.filter(c => c.profile_data?.redrob_signals?.open_to_work_flag)
    }
    if (filter === 'senior') {
      list = list.filter(c => (c.profile_data?.profile?.years_of_experience || 0) >= 7)
    }
    return list
  }, [activeCandidates, query, filter])

  function handleRowClick(c, event) {
    if (event.ctrlKey || event.metaKey) {
      // Compare mode
      setCompareSet(prev => {
        if (prev.find(p => p.profile_data.candidate_id === c.profile_data.candidate_id)) {
          return prev.filter(p => p.profile_data.candidate_id !== c.profile_data.candidate_id)
        }
        if (prev.length >= 2) return [prev[1], c]
        return [...prev, c]
      })
      setView('compare')
    } else {
      setSelected(c)
      setView(v => v === 'compare' ? 'profile' : v)
    }
  }

  const isCompareSelected = (c) =>
    compareSet.some(p => p.profile_data.candidate_id === c.profile_data.candidate_id)

  if (loading) {
    return (
      <div className="layout" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'var(--text-2)', fontSize: '1rem' }}>Loading candidates…</div>
      </div>
    )
  }

  return (
    <div className="layout">
      {/* ── TOPBAR ── */}
      <header className="topbar">
        <div className="topbar-brand">
          <div className="logo-dot" />
          Recruitment Intelligence Dashboard
        </div>
        <div className="topbar-meta">
          {liveResults
            ? <span className="topbar-badge badge-green" style={{ background: 'var(--green-dim)', borderColor: 'rgba(16,185,129,0.4)', color: 'var(--green)' }}>Live Run · {liveResults.length} ranked</span>
            : <span>{candidates.length} candidates ranked</span>
          }
          <span className="topbar-badge">Recruitment Intelligence Platform</span>
        </div>
      </header>

      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-search">
          <div className="search-input-wrap">
            <Search size={14} />
            <input
              className="search-input"
              placeholder="Search by title, company, skill…"
              value={query}
              onChange={e => setQuery(e.target.value)}
            />
          </div>
          <div className="filter-row">
            {['all', 'github', 'open', 'senior'].map(f => (
              <button
                key={f}
                className={`filter-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f === 'all' ? 'All' : f === 'github' ? '⚡ GitHub' : f === 'open' ? '✓ Open' : '7+ YOE'}
              </button>
            ))}
          </div>
        </div>
        <div className="sidebar-count">{filtered.length} showing</div>
        <div className="candidate-list">
          {filtered.map(c => {
            const prof = c.profile_data?.profile || {}
            const isSelected   = selected?.profile_data.candidate_id === c.profile_data.candidate_id
            const isInCompare  = isCompareSelected(c)
            return (
              <div
                key={c.profile_data.candidate_id}
                className={`candidate-row ${isSelected && view !== 'compare' ? 'selected' : ''} ${isInCompare ? 'compare-selected' : ''}`}
                onClick={e => handleRowClick(c, e)}
              >
                <div className="rank-badge">#{c.rank}</div>
                <div className="row-info">
                  <div className="row-title">{prof.current_title}</div>
                  <div className="row-sub">{prof.current_company} · {prof.years_of_experience}yr</div>
                </div>
                <div className="row-score" style={{ color: scoreColor(c.score) }}>
                  {(c.score * 100).toFixed(1)}
                </div>
              </div>
            )
          })}
          {filtered.length === 0 && (
            <div style={{ padding: '24px', color: 'var(--text-3)', fontSize: '0.85rem', textAlign: 'center' }}>
              No candidates match your search
            </div>
          )}
        </div>
      </aside>

      {/* ── MAIN CONTENT ── */}
      <main className="main">
        <nav className="mode-tabs">
          <button className={`mode-tab ${view === 'upload' ? 'active' : ''}`} onClick={() => setView('upload')}>
            <Upload size={15} /> Upload & Rank
          </button>
          <button className={`mode-tab ${view === 'profile' ? 'active' : ''}`} onClick={() => setView('profile')}>
            <User size={15} /> Profile
          </button>
          <button className={`mode-tab ${view === 'heatmap' ? 'active' : ''}`} onClick={() => setView('heatmap')}>
            <LayoutGrid size={15} /> Skill Heatmap
          </button>
          <button className={`mode-tab ${view === 'compare' ? 'active' : ''}`} onClick={() => setView('compare')}>
            <GitCompare size={15} /> Compare {compareSet.length > 0 ? `(${compareSet.length})` : ''}
          </button>
        </nav>
        <div className="content-area">
          {view === 'upload'  && <UploadView onResultsReady={handleResultsReady} />}
          {view === 'profile' && <ProfileView candidate={selected} />}
          {view === 'heatmap' && <HeatmapView candidates={activeCandidates} />}
          {view === 'compare' && <CompareView candidates={compareSet} />}
        </div>
      </main>
    </div>
  )
}
