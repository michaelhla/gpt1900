export const config = { runtime: 'edge' };

export default async function handler(req) {
    if (req.method !== 'POST') {
        return new Response('Method not allowed', { status: 405 });
    }

    const BACKEND_URL = process.env.BACKEND_URL;
    const BACKEND_API_KEY = process.env.BACKEND_API_KEY;

    if (!BACKEND_URL) {
        return new Response(JSON.stringify({ error: 'Backend not configured' }), { status: 500 });
    }

    const { messages, temperature, max_tokens, top_k, conversation_id } = await req.json();

    // Forward real client IP for rate limiting
    const clientIp = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim()
        || req.headers.get('x-real-ip')
        || 'unknown';

    const headers = {
        'Content-Type': 'application/json',
        'X-Forwarded-For': clientIp,
    };
    if (BACKEND_API_KEY) {
        headers['Authorization'] = `Bearer ${BACKEND_API_KEY}`;
    }

    let backendResp;
    try {
        backendResp = await fetch(`${BACKEND_URL}/chat/completions`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ messages, temperature, max_tokens, top_k, conversation_id }),
        });
    } catch (e) {
        return new Response(JSON.stringify({ error: 'Backend unavailable' }), { status: 502 });
    }

    if (!backendResp.ok) {
        const detail = await backendResp.text().catch(() => 'Unknown error');
        return new Response(JSON.stringify({ error: detail }), { status: backendResp.status });
    }

    // Pipe the SSE stream directly back to the client
    return new Response(backendResp.body, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
        },
    });
}
