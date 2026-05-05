package org.rinha;

import io.netty.buffer.ByteBuf;

/**
 * Zero-allocation JSON parser for the /fraud-score request body.
 * Operates directly on the Netty ByteBuf — no String creation, no reflection,
 * no JSON library.
 *
 * Returns a float[14] vector ready for KNN search. Returns EMPTY on malformed input.
 */
final class RequestParser {

    static final float[] EMPTY = new float[0];

    /**
     * Per-I/O-thread output buffer for the parsed feature vector.
     * Allocated once on first use; reused for every request on that thread.
     * Eliminates the only {@code new float[14]} allocation in the hot path.
     */
    static final ThreadLocal<float[]> TL_VEC =
            ThreadLocal.withInitial(() -> new float[VectorStore.DIMS]);

    // ── Public entry point (zero-alloc after first call per thread) ───────────

    /**
     * Parses the JSON body in {@code buf} and writes the 14-dim feature vector
     * into {@code out}. Returns {@code true} on success, {@code false} if the
     * body is missing a required field or is otherwise malformed.
     *
     * <p>The caller owns {@code out} (typically a thread-local). No array is
     * allocated here — the only heap write is 14 float stores into {@code out}.
     */
    static boolean parse(final ByteBuf buf, final float[] out) {
        try {
            doParse(buf, out);
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

    private static void doParse(ByteBuf buf, float[] v) {
        final int base = buf.readerIndex();
        final int end  = base + buf.readableBytes();

        // ── transaction ──────────────────────────────────────────────────────
        final int txOpen  = sectionOpen(buf, base, end, K_TRANSACTION);
        final int txClose = braceClose(buf, txOpen, end);

        final float txAmount     = readFloat(buf, valAt(buf, txOpen, txClose, K_AMOUNT));
        final int   installments = readInt  (buf, valAt(buf, txOpen, txClose, K_INSTALLMENTS));
        // +1 skips the opening '"' of the timestamp string
        final int   reqAtPos     = valAt(buf, txOpen, txClose, K_REQUESTED_AT) + 1;

        // ── customer ─────────────────────────────────────────────────────────
        final int custOpen  = sectionOpen(buf, txClose, end, K_CUSTOMER);
        final int custClose = braceClose(buf, custOpen, end);

        final float custAvg  = readFloat(buf, valAt(buf, custOpen, custClose, K_AVG_AMOUNT));
        final int   txCount  = readInt  (buf, valAt(buf, custOpen, custClose, K_TX_COUNT));
        final int   arrOpen  = arrayOpen(buf, valAt(buf, custOpen, custClose, K_KNOWN_MERCHANTS), custClose);
        final int   arrClose = arrayClose(buf, arrOpen, custClose);

        // ── merchant ─────────────────────────────────────────────────────────
        final int merchOpen  = sectionOpen(buf, custClose, end, K_MERCHANT);
        final int merchClose = braceClose(buf, merchOpen, end);

        // +1 skips the opening '"' of the id string
        final int  midPos  = valAt(buf, merchOpen,  merchClose, K_MERCH_ID) + 1;
        final int  midLen  = stringLen(buf, midPos, merchClose);
        final int  mccPos  = valAt(buf, merchOpen, merchClose, K_MCC) + 1;  // inside quotes
        final int  mcc     = read4Digits(buf, mccPos);
        final float mAmt   = readFloat(buf, valAt(buf, merchOpen, merchClose, K_AVG_AMOUNT));

        // ── terminal ─────────────────────────────────────────────────────────
        final int termOpen  = sectionOpen(buf, merchClose, end, K_TERMINAL);
        final int termClose = braceClose(buf, termOpen, end);

        final boolean isOnline    = readBool(buf, valAt(buf, termOpen, termClose, K_IS_ONLINE));
        final boolean cardPresent = readBool(buf, valAt(buf, termOpen, termClose, K_CARD_PRESENT));
        final float   kmFromHome  = readFloat(buf, valAt(buf, termOpen, termClose, K_KM_FROM_HOME));

        // ── last_transaction ─────────────────────────────────────────────────
        final int   ltKeyPos  = indexOf(buf, termClose, end, K_LAST_TX);
        final int   ltValPos  = skipToValue(buf, ltKeyPos + K_LAST_TX.length, end);
        final boolean hasLast = buf.getByte(ltValPos) != 'n'; // not "null"

        float minutesSinceLast = -1f;
        float kmFromLast       = -1f;

        if (hasLast) {
            final int ltOpen  = sectionOpen2(buf, ltValPos, end);
            final int ltClose = braceClose(buf, ltOpen, end);
            // +1 skips the opening '"' of the timestamp string
            final int   tsPos  = valAt(buf, ltOpen, ltClose, K_TIMESTAMP) + 1;
            final float kmCurr = readFloat(buf, valAt(buf, ltOpen, ltClose, K_KM_FROM_CURRENT));

            final long reqMin  = isoToEpochMin(buf, reqAtPos);
            final long lastMin = isoToEpochMin(buf, tsPos);
            final long diff    = reqMin - lastMin;
            minutesSinceLast   = diff < 0 ? 0f : (float) diff;
            kmFromLast         = kmCurr;
        }

        // ── unknown_merchant ─────────────────────────────────────────────────
        final int unknown = merchantKnown(buf, arrOpen, arrClose, midPos, midLen) ? 0 : 1;

        // ── assemble vector ──────────────────────────────────────────────────
        v[0]  = clamp(txAmount / MAX_AMOUNT);
        v[1]  = clamp(installments / MAX_INSTALL);
        v[2]  = clamp((txAmount / custAvg) / AMT_AVG_RATIO);
        v[3]  = isoHour(buf, reqAtPos) / 23f;
        v[4]  = isoDOW (buf, reqAtPos) / 6f;
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

    // ── Scan helpers ─────────────────────────────────────────────────────────

    /**
     * Naive two-pointer byte search (Boyer-Moore not needed — patterns are short, ~10-20 bytes).
     * Returns absolute ByteBuf index of pattern start, or throws if missing.
     */
    private static int indexOf(ByteBuf buf, int from, int to, byte[] pat) {
        final int pLen = pat.length;
        final int stop = to - pLen;
        outer:
        for (int i = from; i <= stop; i++) {
            for (int j = 0; j < pLen; j++) {
                if (buf.getByte(i + j) != pat[j]) continue outer;
            }
            return i;
        }
        throw new IllegalArgumentException("key not found");
    }

    /** Find '{' for a section key (e.g. "transaction":  {). */
    private static int sectionOpen(ByteBuf buf, int from, int end, byte[] key) {
        int pos = indexOf(buf, from, end, key) + key.length;
        return findChar(buf, pos, end, '{');
    }

    /** Find '{' from an already-known value position (used for last_transaction). */
    private static int sectionOpen2(ByteBuf buf, int from, int end) {
        return findChar(buf, from, end, '{');
    }

    /** Find '[' from an already-known value position. */
    private static int arrayOpen(ByteBuf buf, int from, int end) {
        return findChar(buf, from, end, '[');
    }

    private static int findChar(ByteBuf buf, int from, int end, char c) {
        for (int i = from; i < end; i++) {
            if (buf.getByte(i) == (byte) c) return i;
        }
        throw new IllegalArgumentException("char '" + c + "' not found");
    }

    /** Find the matching '}' for the '{' at openPos. Tracks brace depth. */
    private static int braceClose(ByteBuf buf, int openPos, int end) {
        int depth = 0;
        for (int i = openPos; i < end; i++) {
            byte b = buf.getByte(i);
            if      (b == '{') depth++;
            else if (b == '}') { if (--depth == 0) return i; }
        }
        throw new IllegalArgumentException("unmatched '{'");
    }

    /** Find the matching ']' for the '[' at openPos. */
    private static int arrayClose(ByteBuf buf, int openPos, int end) {
        int depth = 0;
        for (int i = openPos; i < end; i++) {
            byte b = buf.getByte(i);
            if      (b == '[') depth++;
            else if (b == ']') { if (--depth == 0) return i; }
        }
        throw new IllegalArgumentException("unmatched '['");
    }

    /**
     * Find the value position of a key inside [from, to].
     * Returns absolute index of the first non-whitespace byte after ":".
     */
    private static int valAt(ByteBuf buf, int from, int to, byte[] key) {
        int pos = indexOf(buf, from, to, key) + key.length;
        return skipToValue(buf, pos, to);
    }

    /** Skip past ':' and any whitespace; return position of first value byte. */
    private static int skipToValue(ByteBuf buf, int pos, int end) {
        while (pos < end && buf.getByte(pos) != ':') pos++;
        pos++; // skip ':'
        while (pos < end && buf.getByte(pos) <= ' ') pos++;
        return pos;
    }

    // ── Value parsers ────────────────────────────────────────────────────────

    /** Parse floating-point number from ASCII bytes. No String created. */
    private static float readFloat(ByteBuf buf, int pos) {
        boolean neg = buf.getByte(pos) == '-';
        if (neg) pos++;
        long intPart  = 0;
        long fracPart = 0;
        int  fracDiv  = 1;
        boolean frac  = false;
        while (true) {
            final byte b = buf.getByte(pos);
            if (b >= '0' && b <= '9') {
                if (frac) { fracPart = fracPart * 10 + (b - '0'); fracDiv *= 10; }
                else      { intPart  = intPart  * 10 + (b - '0'); }
                pos++;
            } else if (b == '.' && !frac) {
                frac = true; pos++;
            } else {
                break;
            }
        }
        final float r = (float) intPart + (float) fracPart / (float) fracDiv;
        return neg ? -r : r;
    }

    /** Parse unsigned integer from ASCII bytes. */
    private static int readInt(ByteBuf buf, int pos) {
        int r = 0;
        while (true) {
            final byte b = buf.getByte(pos);
            if (b >= '0' && b <= '9') { r = r * 10 + (b - '0'); pos++; }
            else break;
        }
        return r;
    }

    /** Parse JSON boolean (true/false) from first byte. */
    private static boolean readBool(ByteBuf buf, int pos) {
        return buf.getByte(pos) == 't';
    }

    /**
     * Read exactly 4 ASCII digits as an integer (for MCC codes like "5912").
     * pos must point to the first digit (after the opening '"').
     */
    private static int read4Digits(ByteBuf buf, int pos) {
        return (buf.getByte(pos)     - '0') * 1000
             + (buf.getByte(pos + 1) - '0') * 100
             + (buf.getByte(pos + 2) - '0') * 10
             + (buf.getByte(pos + 3) - '0');
    }

    /** Length of a JSON string starting at pos (after opening '"'), up to closing '"'. */
    private static int stringLen(ByteBuf buf, int pos, int end) {
        int len = 0;
        while (pos + len < end && buf.getByte(pos + len) != '"') len++;
        return len;
    }

    // ── ISO-8601 timestamp helpers ────────────────────────────────────────────
    // Format: "YYYY-MM-DDTHH:MM:SSZ"   (pos is byte AFTER the opening '"')
    //          0123456789012345678901
    //                    1111111111

    private static int read2(ByteBuf buf, int pos) {
        return (buf.getByte(pos) - '0') * 10 + (buf.getByte(pos + 1) - '0');
    }

    /** Hour component (0-23) from ISO-8601 timestamp. */
    private static int isoHour(ByteBuf buf, int pos) {
        return read2(buf, pos + 11);
    }

    /**
     * Day of week from ISO-8601 timestamp.
     * Returns 0=Mon … 6=Sun (spec convention).
     * Uses Tomohiko Sakamoto's algorithm — pure integer arithmetic.
     */
    private static int isoDOW(ByteBuf buf, int pos) {
        int y = read2(buf, pos) * 100 + read2(buf, pos + 2); // YYYY
        int m = read2(buf, pos + 5);                          // MM
        int d = read2(buf, pos + 8);                          // DD
        if (m < 3) y--;
        // 0=Sun … 6=Sat
        int dow = (y + y/4 - y/100 + y/400
                + sakamotoT(m) + d) % 7;
        // convert to spec: 0=Mon … 6=Sun
        return (dow + 6) % 7;
    }

    private static int sakamotoT(int m) {
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
    private static long isoToEpochMin(ByteBuf buf, int pos) {
        int y  = read2(buf, pos) * 100 + read2(buf, pos + 2);
        int mo = read2(buf, pos + 5);
        int d  = read2(buf, pos + 8);
        int h  = read2(buf, pos + 11);
        int mi = read2(buf, pos + 14);
        return epochMin(y, mo, d, h, mi);
    }

    private static long epochMin(int y, int mo, int d, int h, int mi) {
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
    private static int monthOffset(int m) {
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
    private static boolean merchantKnown(ByteBuf buf, int arrOpen, int arrClose,
                                         int midPos, int midLen) {
        int i = arrOpen + 1; // skip '['
        while (i < arrClose) {
            final byte b = buf.getByte(i);
            if (b == '"') {
                i++; // skip opening '"'
                if (i + midLen <= arrClose && buf.getByte(i + midLen) == '"') {
                    boolean match = true;
                    for (int j = 0; j < midLen; j++) {
                        if (buf.getByte(i + j) != buf.getByte(midPos + j)) {
                            match = false;
                            break;
                        }
                    }
                    if (match) return true;
                }
                // skip to closing '"' of this entry
                while (i < arrClose && buf.getByte(i) != '"') i++;
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
    private static float mccRisk(int mcc) {
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

    private static float clamp(float v) {
        return v < 0f ? 0f : (v > 1f ? 1f : v);
    }
}

