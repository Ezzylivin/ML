// debug_flow.js
import { io } from "socket.io-client";
import axios from "axios";

const NODE_BACKEND = "https://neov6backend.onrender.com";
const USER_ID = "0x15...acB4"; // Ensure this matches your connected wallet

console.log("🚀 STARTING NEURAL FLOW DIAGNOSTIC...");

// 1. Monitor the Node.js -> Frontend Bridge
const socket = io(NODE_BACKEND, { query: { userId: USER_ID } });

socket.on("connect", () => {
    console.log("✅ STEP 1: Browser Socket connected to Node.js");
    console.log("🕵️ Monitoring for incoming packets...");
});

// 2. Intercept Metric Updates (Frontend perspective)
socket.on("bot_status_update", (data) => {
    console.log("📈 [METRIC PACKET RECEIVED]");
    console.log(`   - Balance: $${data.currentBalance}`);
    console.log(`   - PnL: $${data.unrealizedPnl}`);
    console.log(`   - Positions: ${data.activePositions?.length || 0}`);
});

socket.on("bot_log", (log) => {
    console.log("🧠 [LOG RECEIVED]:", log.message || log);
});

// 🟢 3. FORCE-FEED MODULE: Simulate Python -> Node.js Webhook
const forceFeedMetrics = async () => {
    console.log("\n🧪 ATTEMPTING FORCE-FEED: Sending test metrics to bridge...");
    
    const testPayload = {
        userId: USER_ID,
        type: "bot_status_update",
        data: {
            status: "running",
            currentBalance: 1250.50, // Test value
            unrealizedPnl: 45.20,    // Test value
            exposure: 15,
            activePositions: [{ entry: 65000, size: 0.1, time: new Date() }]
        }
    };

    try {
        const res = await axios.post(`${NODE_BACKEND}/api/internal/broadcast`, testPayload);
        if (res.status === 200) {
            console.log("🚀 FORCE-FEED SUCCESS: Packet sent to Node.js endpoint.");
        }
    } catch (err) {
        console.error("❌ FORCE-FEED FAILED: Node.js endpoint rejected the packet.");
        console.error("   Error:", err.message);
    }
};

// Trigger test packet every 10 seconds
setInterval(forceFeedMetrics, 10000);

socket.on("disconnect", () => console.log("❌ Socket Disconnected"));
