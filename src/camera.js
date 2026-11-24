// Stop streams arriving after cancellation or an unanswered permission prompt.
export async function acquireCamera(getUserMedia, constraints, isCurrent = () => true, timeoutMs = 30000) {
  let expired = false, timer;
  const request = Promise.resolve().then(() => getUserMedia(constraints)).then(stream => {
    if (expired || !isCurrent()) {
      stream.getTracks().forEach(track => track.stop());
      const error = new Error('Camera request cancelled.'); error.name = 'AbortError'; throw error;
    }
    return stream;
  });
  try {
    return await Promise.race([request, new Promise((_, reject) => {
      timer = setTimeout(() => { expired = true; reject(new Error('Camera permission is still pending. Allow camera access for this localhost page in this browser, then start again.')); }, timeoutMs);
    })]);
  } finally { clearTimeout(timer); }
}
