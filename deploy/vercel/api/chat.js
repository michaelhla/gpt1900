export const config = { runtime: 'edge' };

const RUNPOD_BASE = 'https://api.runpod.ai/v2';

export default async function handler(req) {
    if (req.method !== 'POST') {
        return new Response('Method not allowed', { status: 405 });
    }

    const { model, messages, temperature, max_tokens, top_k } = await req.json();

    const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;
    const endpoints = {
        scholar: process.env.BASE_ENDPOINT_ID,
        conversationalist: process.env.COHERENCE_ENDPOINT_ID,
    };

    const endpointId = endpoints[model];
    if (!endpointId) {
        return new Response(JSON.stringify({ error: `Unknown model: ${model}` }), { status: 400 });
    }

    const headers = { Authorization: `Bearer ${RUNPOD_API_KEY}`, 'Content-Type': 'application/json' };

    // Submit job to RunPod
    const runResp = await fetch(`${RUNPOD_BASE}/${endpointId}/run`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ input: { messages, temperature, max_tokens, top_k } }),
    });

    if (!runResp.ok) {
        return new Response(JSON.stringify({ error: 'RunPod submit failed' }), { status: 502 });
    }

    const job = await runResp.json();
    const jobId = job.id;

    // Stream: poll RunPod /stream/{jobId} and convert to SSE
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
        async start(controller) {
            const send = (obj) => controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
            let wakeSent = false;

            try {
                while (true) {
                    const resp = await fetch(`${RUNPOD_BASE}/${endpointId}/stream/${jobId}`, { headers });
                    const data = await resp.json();
                    const status = data.status || '';

                    if (status === 'IN_QUEUE' && !wakeSent) {
                        send({ status: 'waking' });
                        wakeSent = true;
                    }

                    for (const chunk of data.stream || []) {
                        const output = chunk.output;
                        if (output && output.text) {
                            send({ token: output.text });
                        }
                    }

                    if (status === 'COMPLETED') {
                        send({ done: true });
                        break;
                    }
                    if (status === 'FAILED') {
                        send({ error: data.error || 'Generation failed' });
                        break;
                    }

                    await new Promise((r) => setTimeout(r, 250));
                }
            } catch (e) {
                send({ error: e.message });
            } finally {
                controller.close();
            }
        },
    });

    return new Response(stream, {
        headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
        },
    });
}
