// File: debug_python.js
import axios from 'axios';

// 🟢 CONFIG
const PYTHON_URL = "http://74.208.28.77:8000"; // Your VPS IP
const USER_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4"; // Your Wallet Address from the log

const runDebug = async () => {
    console.log(`\n🔍 --- DIAGNOSTIC START: Checking Python Engine at ${PYTHON_URL} ---\n`);

    try {
        // 1. Check Status
        console.log("1️⃣  Pinging Bot Status...");
        const res = await axios.get(`${PYTHON_URL}/api/bot/status`, { 
            params: { userId: USER_ID },
            timeout: 5000 
        });

        const data = res.data;
        console.log(`   ✅ Status Code: ${res.status}`);
        console.log(`   🤖 Bot Status: ${data.status}`);
        console.log(`   💰 Balance: ${data.currentBalance}`);
        console.log(`   🕯️  Candles Count: ${data.candles ? data.candles.length : 0}`);
        console.log(`   📈 Equity Points: ${data.equityCurve ? data.equityCurve.length : 0}`);

        if (data.candles && data.candles.length > 0) {
            console.log("\n   ✅ SAMPLE CANDLE DATA (First Item):");
            console.log(data.candles[0]);
        } else {
            console.error("\n   ❌ CRITICAL FAILURE: Python returned 0 candles.");
            console.log("   👉 This means the issue is in 'main.py' or 'ccxt' fetching.");
        }

    } catch (error) {
        console.error(`\n   ❌ CONNECTION FAILED: ${error.message}`);
        if (error.response) {
            console.error(`   Server Responded: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
        }
    }
    console.log("\n🔍 --- DIAGNOSTIC COMPLETE ---");
};

runDebug();
