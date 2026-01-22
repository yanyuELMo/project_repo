import http from 'k6/http';
import { check, sleep } from 'k6';

// Set APP_URL to your Cloud Run service URL, e.g. https://accident-api-...run.app/predict
const APP_URL = __ENV.APP_URL || '';
if (!APP_URL) {
  throw new Error('Set APP_URL env var to the /predict endpoint before running.');
}

// Set NPZ_PATH to a local npz file with key "frames" [T,H,W,3] uint8.
const NPZ_PATH = __ENV.NPZ_PATH || 'dummy.npz';
const npzFile = http.file(open(NPZ_PATH, 'b'), 'dummy.npz', 'application/octet-stream');

export const options = {
  vus: Number(__ENV.VUS) || 5, // concurrent users
  duration: __ENV.DURATION || '1m',
  thresholds: {
    http_req_duration: ['p(95)<2000'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const res = http.post(APP_URL, { file: npzFile });
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(1);
}
