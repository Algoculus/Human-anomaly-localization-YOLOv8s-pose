import { Routes, Route } from 'react-router-dom'
import HomePage from './pages/HomePage'
import CameraPage from './pages/CameraPage'
import ReceiverPage from './pages/ReceiverPage'
import UploadPage from './pages/UploadPage'
import { Toaster } from './components/ui/toaster'
import { ThemeProvider } from './components/theme-provider'

function App() {
  return (
     <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/receiver" element={<ReceiverPage />} />
        <Route path="/upload" element={<UploadPage />} />
      </Routes>
      <Toaster />
    </ThemeProvider>
  )
}

export default App
