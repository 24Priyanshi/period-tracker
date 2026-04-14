// =============================================================
//  Period Tracker — client-side app (no backend, localStorage only)
// =============================================================

const STORE_KEY = 'periodTracker.v1';

const MONTH_NAMES = ['January','February','March','April','May','June',
                     'July','August','September','October','November','December'];

// ----------------------------------------------------------- state

function loadStore() {
    try {
        const raw = localStorage.getItem(STORE_KEY);
        if (!raw) return emptyStore();
        return { ...emptyStore(), ...JSON.parse(raw) };
    } catch {
        return emptyStore();
    }
}

function emptyStore() {
    return {
        settings: { lastStart: '', cycleLen: 28, periodDur: 5 },
        logs: {},      // { 'YYYY-MM-DD': { flow, cramps, mood, fatigue, bloating, headache } }
        journal: {},   // { 'YYYY-MM-DD': 'text' }
    };
}

function saveStore() {
    localStorage.setItem(STORE_KEY, JSON.stringify(store));
}

const store = loadStore();

// current displayed month
let viewDate = new Date();
viewDate.setDate(1);

// ----------------------------------------------------------- date helpers

const pad = (n) => String(n).padStart(2, '0');
const isoDate = (d) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const parseISO = (s) => {
    if (!s) return null;
    const [y, m, d] = s.split('-').map(Number);
    return new Date(y, m - 1, d);
};
const addDays = (d, n) => {
    const nd = new Date(d);
    nd.setDate(nd.getDate() + n);
    return nd;
};
const daysBetween = (a, b) => Math.round((b - a) / 86400000);
const sameYMD = (a, b) =>
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate();

// ----------------------------------------------------------- prediction

/**
 * Compute predicted period ranges for the next few cycles and the fertile
 * window. Returns sets of ISO-date strings keyed by role.
 */
function computePredictions() {
    const { lastStart, cycleLen, periodDur } = store.settings;
    const start = parseISO(lastStart);
    if (!start) return { past: new Set(), predicted: new Set(), fertile: new Set() };

    const past = new Set();
    const predicted = new Set();
    const fertile = new Set();

    // Mark the most recent period (past)
    for (let i = 0; i < periodDur; i++) {
        past.add(isoDate(addDays(start, i)));
    }

    // Project 6 cycles forward
    for (let k = 1; k <= 6; k++) {
        const nextStart = addDays(start, cycleLen * k);
        for (let i = 0; i < periodDur; i++) {
            predicted.add(isoDate(addDays(nextStart, i)));
        }
        // fertile window ~ days 10 to 16 before next period (classic estimate)
        for (let i = 10; i <= 16; i++) {
            fertile.add(isoDate(addDays(nextStart, -i)));
        }
    }

    return { past, predicted, fertile };
}

function updatePredictionBox() {
    const box = document.getElementById('predictionBox');
    const { lastStart, cycleLen, periodDur } = store.settings;
    const start = parseISO(lastStart);
    if (!start) {
        box.innerHTML = '<p class="muted">Enter your last period date to see predictions.</p>';
        return;
    }

    const nextStart = addDays(start, cycleLen);
    const nextEnd = addDays(nextStart, periodDur - 1);
    const ovStart = addDays(nextStart, -16);
    const ovEnd = addDays(nextStart, -10);

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const daysUntil = daysBetween(today, nextStart);

    const fmt = (d) => d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const untilTxt = daysUntil === 0 ? 'today'
                    : daysUntil > 0  ? `in ${daysUntil} day${daysUntil === 1 ? '' : 's'}`
                    : `${-daysUntil} day${daysUntil === -1 ? '' : 's'} ago`;

    box.innerHTML = `
        <p><strong>Next period:</strong><br>
           ${fmt(nextStart)} – ${fmt(nextEnd)} (${untilTxt})</p>
        <p style="margin-top:0.6rem"><strong>Fertile window:</strong><br>
           ${fmt(ovStart)} – ${fmt(ovEnd)}</p>
        <p class="muted" style="margin-top:0.6rem; font-size:12px">
           Based on a ${cycleLen}-day cycle &amp; ${periodDur}-day period.</p>
    `;
}

// ----------------------------------------------------------- calendar render

function renderCalendar() {
    const year = viewDate.getFullYear();
    const month = viewDate.getMonth();
    document.getElementById('monthLabel').textContent =
        `${MONTH_NAMES[month]} ${year}`;

    const cal = document.getElementById('calendar');
    cal.innerHTML = '';

    // day-name header
    ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].forEach(d => {
        const el = document.createElement('div');
        el.className = 'day';
        el.textContent = d;
        cal.appendChild(el);
    });

    const firstWeekday = new Date(year, month, 1).getDay();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const daysInPrevMonth = new Date(year, month, 0).getDate();

    // prev-month filler
    for (let i = firstWeekday - 1; i >= 0; i--) {
        const el = document.createElement('div');
        el.className = 'prev-date';
        el.textContent = daysInPrevMonth - i;
        cal.appendChild(el);
    }

    const { past, predicted, fertile } = computePredictions();
    const today = new Date();

    // current-month days
    for (let d = 1; d <= daysInMonth; d++) {
        const el = document.createElement('div');
        el.className = 'number';
        el.textContent = d;

        const iso = isoDate(new Date(year, month, d));
        if (past.has(iso))      el.classList.add('period');
        if (predicted.has(iso)) el.classList.add('predicted');
        if (fertile.has(iso))   el.classList.add('fertility');
        if (sameYMD(today, new Date(year, month, d))) el.classList.add('today');
        if (store.logs[iso])    el.classList.add('has-log');

        el.title = iso;
        el.addEventListener('click', () => {
            // jump to tracker for this day
            document.getElementById('trackerDateInput').value = iso;
            switchPage('tracker');
            loadLogIntoTracker(iso);
        });

        cal.appendChild(el);
    }

    // trailing filler to complete final week
    const used = firstWeekday + daysInMonth;
    const trailing = (7 - (used % 7)) % 7;
    for (let i = 1; i <= trailing; i++) {
        const el = document.createElement('div');
        el.className = 'prev-date';
        el.textContent = i;
        cal.appendChild(el);
    }
}

// ----------------------------------------------------------- nav

function switchPage(name) {
    document.querySelectorAll('.page').forEach(p =>
        p.classList.toggle('hidden', p.id !== `page-${name}`));
    document.querySelectorAll('.poppy').forEach(li =>
        li.classList.toggle('active', li.dataset.page === name));
    if (name === 'home') renderCalendar();
    if (name === 'journal') renderJournalList();
}

// ----------------------------------------------------------- settings form

function wireSettings() {
    const lastInput = document.getElementById('lastStart');
    const cycleInput = document.getElementById('cycleLen');
    const durInput = document.getElementById('periodDur');

    lastInput.value = store.settings.lastStart || '';
    cycleInput.value = store.settings.cycleLen;
    durInput.value = store.settings.periodDur;

    document.getElementById('saveSettings').addEventListener('click', () => {
        store.settings.lastStart = lastInput.value;
        store.settings.cycleLen = parseInt(cycleInput.value, 10) || 28;
        store.settings.periodDur = parseInt(durInput.value, 10) || 5;
        saveStore();
        updatePredictionBox();
        renderCalendar();
    });
}

// ----------------------------------------------------------- tracker

function loadLogIntoTracker(iso) {
    const log = store.logs[iso] || {};
    document.querySelectorAll('.likert').forEach(row => {
        const field = row.dataset.field;
        row.querySelectorAll('button').forEach(btn => {
            btn.classList.toggle('selected',
                String(log[field]) === btn.dataset.v);
        });
    });
    document.getElementById('trackerDate').textContent = iso;
    document.getElementById('trackerStatus').textContent = '';
}

function wireTracker() {
    const dateInput = document.getElementById('trackerDateInput');
    const today = isoDate(new Date());
    dateInput.value = today;
    document.getElementById('trackerDate').textContent = today;

    dateInput.addEventListener('change', () => {
        loadLogIntoTracker(dateInput.value);
    });

    // likert button toggling
    document.querySelectorAll('.likert').forEach(row => {
        row.addEventListener('click', e => {
            if (e.target.tagName !== 'BUTTON') return;
            row.querySelectorAll('button').forEach(b =>
                b.classList.toggle('selected', b === e.target));
        });
    });

    document.getElementById('saveLog').addEventListener('click', () => {
        const iso = dateInput.value;
        if (!iso) return;
        const entry = {};
        document.querySelectorAll('.likert').forEach(row => {
            const sel = row.querySelector('button.selected');
            if (sel) entry[row.dataset.field] = parseInt(sel.dataset.v, 10);
        });
        if (Object.keys(entry).length === 0) {
            delete store.logs[iso];
        } else {
            store.logs[iso] = entry;
        }
        saveStore();
        document.getElementById('trackerStatus').textContent =
            `Saved log for ${iso}`;
        renderCalendar();
    });

    loadLogIntoTracker(today);
}

// ----------------------------------------------------------- journal

function wireJournal() {
    const dateInput = document.getElementById('journalDate');
    const txt = document.getElementById('journalText');
    const today = isoDate(new Date());
    dateInput.value = today;
    txt.value = store.journal[today] || '';

    dateInput.addEventListener('change', () => {
        txt.value = store.journal[dateInput.value] || '';
    });

    document.getElementById('saveJournal').addEventListener('click', () => {
        const iso = dateInput.value;
        if (!iso) return;
        const value = txt.value.trim();
        if (value) {
            store.journal[iso] = value;
        } else {
            delete store.journal[iso];
        }
        saveStore();
        document.getElementById('journalStatus').textContent =
            value ? `Saved entry for ${iso}` : `Cleared entry for ${iso}`;
        renderJournalList();
    });

    renderJournalList();
}

function renderJournalList() {
    const list = document.getElementById('journalList');
    if (!list) return;
    const entries = Object.entries(store.journal).sort(([a], [b]) => b.localeCompare(a));
    if (entries.length === 0) {
        list.innerHTML = '<p class="muted">No entries yet.</p>';
        return;
    }
    list.innerHTML = entries.slice(0, 8).map(([iso, text]) => `
        <div class="entry">
            <time>${iso}</time>
            <p>${escapeHtml(text)}</p>
        </div>
    `).join('');
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
    }[c]));
}

// ----------------------------------------------------------- init

function init() {
    // nav
    document.querySelectorAll('.poppy').forEach(li =>
        li.addEventListener('click', () => switchPage(li.dataset.page)));

    // month navigation
    document.getElementById('prevMonth').addEventListener('click', () => {
        viewDate.setMonth(viewDate.getMonth() - 1);
        renderCalendar();
    });
    document.getElementById('nextMonth').addEventListener('click', () => {
        viewDate.setMonth(viewDate.getMonth() + 1);
        renderCalendar();
    });

    wireSettings();
    wireTracker();
    wireJournal();

    switchPage('home');
    updatePredictionBox();
}

document.addEventListener('DOMContentLoaded', init);
