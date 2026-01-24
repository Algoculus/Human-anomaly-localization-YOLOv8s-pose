import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import CameraPage from './pages/CameraPage'
import ReceiverPage from './pages/ReceiverPage'
import UploadPage from './pages/UploadPage'
import { Toaster } from './components/ui/toaster'

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/receiver" element={<ReceiverPage />} />
        <Route path="/upload" element={<UploadPage />} />
      </Routes>
      <Toaster />
    </>
  )
}

export default App
