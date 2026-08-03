import axios from 'axios';
import { PredictResponsePayload, BatchScreenResponse, CandidateItem } from '../types';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const api = {
  async predictSingle(cifText?: string, file?: File): Promise<PredictResponsePayload> {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else if (cifText) {
      formData.append('cif_text', cifText);
    } else {
      throw new Error('Either file or cifText must be provided.');
    }

    const res = await axios.post<PredictResponsePayload>(`${API_BASE_URL}/api/predict`, formData);
    return res.data;
  },

  async screenBatch(files: File[], confidenceFilter = 'all'): Promise<BatchScreenResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    formData.append('confidence_filter', confidenceFilter);

    const res = await axios.post<BatchScreenResponse>(`${API_BASE_URL}/api/screen`, formData);
    return res.data;
  },

  async searchMaterials(formula = '', element = '') {
    const res = await axios.get(`${API_BASE_URL}/api/search`, {
      params: { formula, element },
    });
    return res.data;
  },

  async screenSearchedMaterial(materialId: string): Promise<PredictResponsePayload> {
    const res = await axios.get<PredictResponsePayload>(`${API_BASE_URL}/api/search/${materialId}/screen`);
    return res.data;
  },

  async compareModels(cifText?: string, file?: File) {
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    } else if (cifText) {
      formData.append('cif_text', cifText);
    }

    const res = await axios.post(`${API_BASE_URL}/api/compare`, formData);
    return res.data;
  },

  async fetchHistory() {
    const res = await axios.get(`${API_BASE_URL}/api/history`);
    return res.data;
  },

  async downloadCSV(candidates: CandidateItem[]) {
    const res = await axios.post(`${API_BASE_URL}/api/report/csv`, { candidates }, {
      responseType: 'blob',
    });
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'MatScreen_AI_Candidates.csv');
    document.body.appendChild(link);
    link.click();
    link.remove();
  },
};
