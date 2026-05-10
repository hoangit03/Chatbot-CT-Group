import axios from 'axios';

// Gateway to local Nginx which will proxy /api/v1 to the backend
const GATEWAY_URL = '/api/v1';

const apiClient = axios.create({
  baseURL: GATEWAY_URL,
});

export const api = {
  // Projects
  getProjects: async () => {
    // Mock projects for HR Chatbot
    return { projects: ['Quy Định Cty', 'Sổ Tay Nhân Viên'] };
  },

  // ETL
  uploadETL: async (tenantId, file, minRoleLevel, forceOverwrite = false, projectName = "") => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('min_role_level', minRoleLevel);
    formData.append('force_overwrite', forceOverwrite);
    if (projectName) {
      formData.append('project_name', projectName);
    }
    
    try {
      // Assuming Nginx proxies /api/v1/etl/extract to the global ETL service on port 8001
      const { data } = await apiClient.post(`/etl/extract`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      return data;
    } catch (error) {
      const errBody = error.response?.data;
      if (errBody?.detail?.code === "CONFIRM_OVERWRITE") {
        throw new Error("CONFIRM_OVERWRITE");
      }
      throw new Error(errBody?.detail?.message || errBody?.detail || 'Lỗi upload dữ liệu');
    }
  },

  getETLFiles: async (tenantId) => {
    return { files: [] };
  },

  // Chat/Sessions 
  getSessions: async (tenantId, page = 1, limit = 20) => {
    return { sessions: [], total: 0 };
  },
  getSessionMessages: async (tenantId, sessionId) => {
    return { messages: [] };
  },

  // Send message
  sendMessage: async (tenantId, sessionId, message) => {
    const payload = {
      message: message,
      session_id: sessionId,
      project_name: tenantId
    };
    const { data } = await apiClient.post('/chat', payload);
    return data;
  }
};

