// File: /root/Project/ML/checkSchema.js
import mongoose from 'mongoose';
import axios from 'axios';

// 🟢 CONFIG - USE YOUR ACTUAL MONGO URI
const MONGO_URI = "mongodb+srv://..."; 
const WALLET_ID = "0x1589DB9ef013Bb3394089191E5b76E49af1AacB4";
const PYTHON_URL = `http://localhost:8000/api/bot/status?userId=${WALLET_ID}`;

// This mimics the Backend Schema to see where it breaks
const BotSchema = new mongoose.Schema({
    userId: String,
    status: String,
    candles: Array,
    logs: Array,
    currentBalance: Number
}, { strict: false });

const BotTest = mongoose.model('BotTest', BotSchema);

async function run() {
    console.log("🚀 Starting Diagnostic on ML Server...");
    try {
        await mongoose.connect(MONGO_URI);
        console.log("✅ Connected to MongoDB.");

        const res = await axios.get(PYTHON_URL);
        const data = res.data;

        if (!data || data.status === "stopped") {
            console.error("❌ main3.py is reporting 'stopped'. Start the bot in the UI first.");
            return;
        }

        console.log("🧪 testing data serialization...");
        const test = new BotTest({
            userId: WALLET_ID,
            status: data.status,
            candles: data.candles,
            logs: data.logs,
            currentBalance: data.currentBalance
        });

        // This is the moment of truth: MongoDB cannot store NaN or Infinity
        await test.save();
        console.log("✅ SUCCESS: Database accepted the data.");
        await BotTest.deleteOne({ _id: test._id });

    } catch (err) {
        console.log("\n❌ DIAGNOSTIC FOUND THE ISSUE:");
        console.log(err.message);
        
        if (err.message.includes("NaN")) {
            console.log("\n👉 CAUSE: Your Python indicators are producing 'NaN' values.");
            console.log("👉 FIX: Add '.fillna(0)' to your indicators in main3.py.");
        }
    } finally {
        mongoose.connection.close();
    }
}
run();
