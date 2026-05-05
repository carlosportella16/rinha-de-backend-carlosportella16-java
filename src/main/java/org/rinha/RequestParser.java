package org.rinha;

import io.netty.buffer.ByteBuf;

/**
 * Zero-allocation JSON parser for the /fraud-score request body.
 *
 * HOT PATH STRATEGY:
 *   buf.getBytes() copies the request body ONCE into a thread-local byte[].
 *   All subsequent field access uses direct byte[] array loads — no virtual
 *   dispatch, no polymorphic inline-cache misses on getByte().
 *
 *   Cost:  1× bulk copy of ~500 bytes  (~40 ns via arraycopy/memcpy)
 *   Saves: ~200× buf.getByte() virtual calls  (~1000–2000 ns total)
 *   Net:   ~5–15× faster parse path after JIT warmup.
 */
final class RequestParser {

    static final float[] EMPTY = new float[0];

    /**
     * Per-I/O-thread output buffer for the parsed feature vector.
     * Allocated once on first use; reused for every request on that thread.
     */
    static final ThreadLocal<float[]> TL_VEC =
            ThreadLocal.withInitial(() -> new float[VectorStore.DIMS]);

    /**
     * Per-I/O-thread raw body buffer.
     * Copied from ByteBuf once per request; grows on demand (rare for ~500-byte bodies).
     */
    private static final ThreadLocal<byte[]> TL_BYTES =
            ThreadLocal.withInitial(() -> new byte[2048]);

    // ── Public entry point ────────────────────────────────────────────────────

    /**
     * Copies {@code buf} into a thread-local byte[], then parses the 14-dim
     * feature vector into {@code out}. Returns {@code true} on success,
     * {@code false} if the body is malformed.
     */
    static boolean parse(final ByteBuf buf, final float[] out) {
        final int len = buf.readableBytes();
        byte[] b = TL_BYTES.get();
        if (len > b.length) {
            b = new byte[(len + 511) & ~511];   // round up to 512-byte boundary
            TL_BYTES.set(b);
        }
        // Single bulk copy: avoids ~200+ virtual getByte() calls in the hot path
        buf.getBytes(buf.readerIndex(), b, 0, len);
        try {
            doParseBytes(b, len, out);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /** Legacy overload retained for tests; allocates a fresh float[14]. */
    static float[] parse(final ByteBuf buf) {
        final float[] v = new float[VectorStore.DIMS];
        return parse(buf, v) ? v : EMPTY;
    }

    private static final float MAX_AMOUNT    = 10_000f;
    private static final float MAX_INSTALL   = 12f;
    private static final float AMT_AVG_RATIO = 10f;
    private static final float MAX_MINUTES   = 1_440f;
    private static final float MAX_KM        = 1_000f;
    private static final float MAX_TX_24H    = 20f;
    private static final float MAX_MERCH_AMT = 10_000f;

    // ── Pre-encoded search keys (ASCII — all JSON field names are ASCII) ───────

    private static final byte[] K_TRANSACTION     = k("\"transaction\"");
    private static final byte[] K_AMOUNT          = k("\"amount\"");
    private static final byte[] K_INSTALLMENTS    = k("\"installments\"");
    private static final byte[] K_REQUESTED_AT    = k("\"requested_at\"");
    private static final byte[] K_CUSTOMER        = k("\"customer\"");
    private static final byte[] K_AVG_AMOUNT      = k("\"avg_amount\"");
    private static final byte[] K_TX_COUNT        = k("\"tx_count_24h\"");
    private static final byte[] K_KNOWN_MERCHANTS = k("\"known_merchants\"");
    private static final byte[] K_MERCHANT        = k("\"merchant\"");
    private static final byte[] K_MERCH_ID        = k("\"id\"");
    private static final byte[] K_MCC             = k("\"mcc\"");
    private static final byte[] K_TERMINAL        = k("\"terminal\"");
    private static final byte[] K_IS_ONLINE       = k("\"is_online\"");
    private static final byte[] K_CARD_PRESENT    = k("\"card_present\"");
    private static final byte[] K_KM_FROM_HOME    = k("\"km_from_home\"");
    private static final byte[] K_LAST_TX         = k("\"last_transaction\"");
    private static final byte[] K_TIMESTAMP       = k("\"timestamp\"");
    private static final byte[] K_KM_FROM_CURRENT = k("\"km_from_current\"");

    private static byte[] k(String s) {
        byte[] b = new byte[s.length()];
        for (int i = 0; i < s.length(); i++) b[i] = (byte) s.charAt(i);
        return b;
    }

    // ── Core parser — operates on raw byte[] (no ByteBuf in hot path) ─────────

    private static void doParseBytes(final byte[] b, final int end, final float[] v) {

        // ── transaction ──────────────────────────────────────────────────────
        final int txOpen  = sectionOpen(b, 0, end, K_TRANSACTION);
        final int txClose = braceClose(b, txOpen, end);

        final float txAmount     = readFloat(b, valAt(b, txOpen, txClose, K_AMOUNT));
        final int   installments = readInt  (b, valAt(b, txOpen, txClose, K_INSTALLMENTS));
        // +1 skips the opening '"' of the timestamp string
        final int   reqAtPos     = valAt(b, txOpen, txClose, K_REQUESTED_AT) + 1;

        // ── customer ─────────────────────────────────────────────────────────
        final int custOpen  = sectionOpen(b, txClose, end, K_CUSTOMER);
        final int custClose = braceClose(b, custOpen, end);

        final float custAvg  = readFloat(b, valAt(b, custOpen, custClose, K_AVG_AMOUNT));
        final int   txCount  = readInt  (b, valAt(b, custOpen, custClose, K_TX_COUNT));
        final int   arrOpen  = arrayOpen(b, valAt(b, custOpen, custClose, K_KNOWN_MERCHANTS), custClose);
        final int   arrClose = arrayClose(b, arrOpen, custClose);

        // ── merchant ─────────────────────────────────────────────────────────
        final int merchOpen  = sectionOpen(b, custClose, end, K_MERCHANT);
        final int merchClose = braceClose(b, merchOpen, end);

        // +1 skips the opening '"' of the id string
        final int   midPos  = valAt(b, merchOpen, merchClose, K_MERCH_ID) + 1;
        final int   midLen  = stringLen(b, midPos, merchClose);
        final int   mccPos  = valAt(b, merchOpen, merchClose, K_MCC) + 1;
        final int   mcc     = read4Digits(b, mccPos);
        final float mAmt    = readFloat(b, valAt(b, merchOpen, merchClose, K_AVG_AMOUNT));

        // ── terminal ─────────────────────────────────────────────────────────
        final int termOpen  = sectionOpen(b, merchClose, end, K_TERMINAL);
        final int termClose = braceClose(b, termOpen, end);

        final boolean isOnline    = readBool(b, valAt(b, termOpen, termClose, K_IS_ONLINE));
        final boolean cardPresent = readBool(b, valAt(b, termOpen, termClose, K_CARD_PRESENT));
        final float   kmFromHome  = readFloat(b, valAt(b, termOpen, termClose, K_KM_FROM_HOME));

        // ── last_transaction ─────────────────────────────────────────────────
        final int     ltKeyPos = indexOf(b, termClose, end, K_LAST_TX);
        final int     ltValPos = skipToValue(b, ltKeyPos + K_LAST_TX.length, end);
        final boolean hasLast  = b[ltValPos] != 'n'; // not "null"

        float minutesSinceLast = -1f;
        float kmFromLast       = -1f;

        if (hasLast) {
            final int ltOpen  = sectionOpen2(b, ltValPos, end);
            final int ltClose = braceClose(b, ltOpen, end);
            // +1 skips the opening '"' of the timestamp string
            final int   tsPos  = valAt(b, ltOpen, ltClose, K_TIMESTAMP) + 1;
            final float kmCurr = readFloat(b, valAt(b, ltOpen, ltClose, K_KM_FROM_CURRENT));

            final long reqMin  = isoToEpochMin(b, reqAtPos);
            final long lastMin = isoToEpochMin(b, tsPos);
            final long diff    = reqMin - lastMin;
            minutesSinceLast   = diff < 0 ? 0f : (float) diff;
            kmFromLast         = kmCurr;
        }

        // ── unknown_merchant ─────────────────────────────────────────────────
        final int unknown = merchantKnown(b, arrOpen, arrClose, midPos, midLen) ? 0 : 1;

        // ── assemble 14-dim vector ────────────────────────────────────────────
        v[0]  = clamp(txAmount / MAX_AMOUNT);
        v[1]  = clamp(installments / MAX_INSTALL);
        v[2]  = clamp((txAmount / custAvg) / AMT_AVG_RATIO);
        v[3]  = isoHour(b, reqAtPos) / 23f;
        v[4]  = isoDOW (b, reqAtPos) / 6f;
        v[5]  = hasLast ? clamp(minutesSinceLast / MAX_MINUTES) : -1f;
        v[6]  = hasLast ? clamp(kmFromLast       / MAX_KM)      : -1f;
        v[7]  = clamp(kmFromHome / MAX_KM);
        v[8]  = clamp(txCount    / MAX_TX_24H);
        v[9]  = isOnline    ? 1f : 0f;
        v[10] = cardPresent ? 1f : 0f;
        v[11] = unknown;
        v[12] = mccRisk(mcc);
        v[13] = clamp(mAmt / MAX_MERCH_AMT);
    }

    // ── Scan helpers — all on byte[] (no ByteBuf in hot path) ───────────────

    /**
     * First-byte filter + full scan. Avoids inner comparison for most positions.
     * For JSON keys (~10–20 byte patterns), the first '"' byte filters 98%+ of
     * positions before any multi-byte comparison is attempted.
     */
    private static int indexOf(final byte[] b, final int from, final int to, final byte[] pat) {
        final int  pLen  = pat.length;
        final int  stop  = to - pLen;
        final byte first = pat[0];
        outer:
        for (int i = from; i <= stop; i++) {
            if (b[i] == first) {
                for (int j = 1; j < pLen; j++) {
                    if (b[i + j] != pat[j]) continue outer;
                }
                return i;
            }
        }
        throw new IllegalArgumentException("key not found");
    }

    /** Find '{' for a section key (e.g. "transaction":  {). */
    private static int sectionOpen(final byte[] b, final int from, final int end, final byte[] key) {
        return findChar(b, indexOf(b, from, end, key) + key.length, end, (byte) '{');
    }

    /** Find '{' from an already-known value position (used for last_transaction). */
    private static int sectionOpen2(final byte[] b, final int from, final int end) {
        return findChar(b, from, end, (byte) '{');
    }

    /** Find '[' from an already-known value position. */
    private static int arrayOpen(final byte[] b, final int from, final int end) {
        return findChar(b, from, end, (byte) '[');
    }

    private static int findChar(final byte[] b, final int from, final int end, final byte c) {
        for (int i = from; i < end; i++) {
            if (b[i] == c) return i;
        }
        throw new IllegalArgumentException("char not found");
    }

    /** Find the matching '}' for the '{' at openPos. Tracks brace depth. */
    private static int braceClose(final byte[] b, final int openPos, final int end) {
        int depth = 0;
        for (int i = openPos; i < end; i++) {
            final byte c = b[i];
            if      (c == '{') depth++;
            else if (c == '}') { if (--depth == 0) return i; }
        }
        throw new IllegalArgumentException("unmatched '{'");
    }

    /** Find the matching ']' for the '[' at openPos. */
    private static int arrayClose(final byte[] b, final int openPos, final int end) {
        int depth = 0;
        for (int i = openPos; i < end; i++) {
            final byte c = b[i];
            if      (c == '[') depth++;
            else if (c == ']') { if (--depth == 0) return i; }
        }
        throw new IllegalArgumentException("unmatched '['");
    }

    /**
     * Find the value position of a key inside [from, to].
     * Returns absolute index of the first non-whitespace byte after ":".
     */
    private static int valAt(final byte[] b, final int from, final int to, final byte[] key) {
        return skipToValue(b, indexOf(b, from, to, key) + key.length, to);
    }

    /** Skip past ':' and any whitespace; return position of first value byte. */
    private static int skipToValue(final byte[] b, int pos, final int end) {
        while (pos < end && b[pos] != ':') pos++;
        pos++; // skip ':'
        while (pos < end && b[pos] <= ' ') pos++;
        return pos;
    }

    // ── Value parsers — byte[] based, int arithmetic (avoids long where possible) ──

    /**
     * Parses a floating-point number from ASCII bytes. Uses int arithmetic —
     * all values in this domain (amount ≤ 10000, km ≤ 1000) fit in int.
     */
    private static float readFloat(final byte[] b, int pos) {
        final boolean neg = b[pos] == '-';
        if (neg) pos++;
        int intPart = 0, fracPart = 0, fracDiv = 1;
        boolean frac = false;
        byte c;
        while (true) {
            c = b[pos];
            if (c >= '0' && c <= '9') {
                if (!frac) { intPart = intPart * 10 + (c - '0'); }
                else       { fracPart = fracPart * 10 + (c - '0'); fracDiv *= 10; }
                pos++;
            } else if (c == '.' && !frac) {
                frac = true; pos++;
            } else {
                break;
            }
        }
        final float r = intPart + (float) fracPart / fracDiv;
        return neg ? -r : r;
    }

    /** Parse unsigned integer from ASCII bytes. */
    private static int readInt(final byte[] b, int pos) {
        int r = 0;
        byte c;
        while ((c = b[pos]) >= '0' && c <= '9') { r = r * 10 + (c - '0'); pos++; }
        return r;
    }

    /** Parse JSON boolean (true/false) from first byte. */
    private static boolean readBool(final byte[] b, final int pos) {
        return b[pos] == 't';
    }

    /**
     * Read exactly 4 ASCII digits as an integer (for MCC codes like "5912").
     * pos must point to the first digit (after the opening '"').
     */
    private static int read4Digits(final byte[] b, final int pos) {
        return (b[pos] - '0') * 1000
             + (b[pos+1] - '0') * 100
             + (b[pos+2] - '0') * 10
             + (b[pos+3] - '0');
    }

    /** Length of a JSON string starting at pos (after opening '"'), up to closing '"'. */
    private static int stringLen(final byte[] b, final int pos, final int end) {
        int len = 0;
        while (pos + len < end && b[pos + len] != '"') len++;
        return len;
    }

    // ── ISO-8601 timestamp helpers ────────────────────────────────────────────
    // Format: "YYYY-MM-DDTHH:MM:SSZ"   (pos is byte AFTER the opening '"')
    //          0123456789012345678901
    //                    1111111111

    private static int read2(final byte[] b, final int pos) {
        return (b[pos] - '0') * 10 + (b[pos+1] - '0');
    }

    /** Hour component (0-23) from ISO-8601 timestamp. */
    private static int isoHour(final byte[] b, final int pos) {
        return read2(b, pos + 11);
    }

    /**
     * Day of week from ISO-8601 timestamp.
     * Returns 0=Mon … 6=Sun (spec convention).
     * Uses Tomohiko Sakamoto's algorithm — pure integer arithmetic.
     */
    private static int isoDOW(final byte[] b, final int pos) {
        int y = read2(b, pos) * 100 + read2(b, pos + 2); // YYYY
        int m = read2(b, pos + 5);                        // MM
        int d = read2(b, pos + 8);                        // DD
        if (m < 3) y--;
        // 0=Sun … 6=Sat
        int dow = (y + y/4 - y/100 + y/400
                + sakamotoT(m) + d) % 7;
        // convert to spec: 0=Mon … 6=Sun
        return (dow + 6) % 7;
    }

    private static int sakamotoT(final int m) {
        // t[] = {0,3,2,5,0,3,5,1,4,6,2,4} — Tomohiko Sakamoto's offset table
        switch (m) {
            case  1: return 0;  case  2: return 3;  case  3: return 2;
            case  4: return 5;  case  5: return 0;  case  6: return 3;
            case  7: return 5;  case  8: return 1;  case  9: return 4;
            case 10: return 6;  case 11: return 2;  default: return 4; // 12
        }
    }

    /**
     * Convert ISO-8601 timestamp to epoch minutes (UTC).
     * No Date/Instant allocation; runs on raw bytes.
     */
    private static long isoToEpochMin(final byte[] b, final int pos) {
        final int y  = read2(b, pos) * 100 + read2(b, pos + 2);
        final int mo = read2(b, pos + 5);
        final int d  = read2(b, pos + 8);
        final int h  = read2(b, pos + 11);
        final int mi = read2(b, pos + 14);
        return epochMin(y, mo, d, h, mi);
    }

    private static long epochMin(final int y, final int mo, final int d,
                                 final int h, final int mi) {
        // Leap year
        final boolean leap = (y % 4 == 0) && (y % 100 != 0 || y % 400 == 0);
        // Day of year (0-based)
        int doy = monthOffset(mo) + d - 1;
        if (mo > 2 && leap) doy++;
        // Leap years before y: floor((y-1)/4) - floor((y-1)/100) + floor((y-1)/400)
        final int yy     = y - 1;
        final int leaps  = yy / 4 - yy / 100 + yy / 400 - 477; // −477 = base at y=1970
        final int days   = (y - 1970) * 365 + leaps + doy;
        return (long) days * 1440L + h * 60L + mi;
    }

    /** Day-of-year offset for month m (1-12), non-leap-year. */
    private static int monthOffset(final int m) {
        switch (m) {
            case  1: return   0; case  2: return  31; case  3: return  59;
            case  4: return  90; case  5: return 120; case  6: return 151;
            case  7: return 181; case  8: return 212; case  9: return 243;
            case 10: return 273; case 11: return 304; default: return 334; // 12
        }
    }

    // ── Known-merchant membership check ─────────────────────────────────────

    /**
     * Returns true if the merchant id (at midPos, length midLen) appears anywhere
     * inside the known_merchants array bytes [arrOpen+1 … arrClose-1].
     * Byte-by-byte comparison — no String allocation.
     */
    private static boolean merchantKnown(final byte[] b, final int arrOpen, final int arrClose,
                                         final int midPos, final int midLen) {
        int i = arrOpen + 1; // skip '['
        while (i < arrClose) {
            if (b[i] == '"') {
                i++; // skip opening '"'
                if (i + midLen <= arrClose && b[i + midLen] == '"') {
                    boolean match = true;
                    for (int j = 0; j < midLen; j++) {
                        if (b[i + j] != b[midPos + j]) { match = false; break; }
                    }
                    if (match) return true;
                }
                // skip to closing '"' of this entry
                while (i < arrClose && b[i] != '"') i++;
            }
            i++;
        }
        return false;
    }

    // ── MCC risk lookup ──────────────────────────────────────────────────────

    /**
     * Hard-coded from mcc_risk.json — file never changes during the competition.
     * Operates on 4-digit int — no string allocation.
     */
    private static float mccRisk(final int mcc) {
        switch (mcc) {
            case 5411: return 0.15f;
            case 5812: return 0.30f;
            case 5912: return 0.20f;
            case 5944: return 0.45f;
            case 7801: return 0.80f;
            case 7802: return 0.75f;
            case 7995: return 0.85f;
            case 4511: return 0.35f;
            case 5311: return 0.25f;
            case 5999: return 0.50f;
            default:   return 0.50f;
        }
    }

    // ── Math ─────────────────────────────────────────────────────────────────

    private static float clamp(final float v) {
        return v < 0f ? 0f : (v > 1f ? 1f : v);
    }
}
