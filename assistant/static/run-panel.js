// FEA Workbench — manual engine operation inside the unified app.
// Drives POST /api/run (no LLM required), renders a results card, and the
// ingested report is opened in the wiki view on demand.
document.addEventListener('DOMContentLoaded', () => {
    const tabWiki = document.getElementById('tab-wiki');
    const tabAnalysis = document.getElementById('tab-analysis');
    const wikiContent = document.getElementById('wiki-content');
    const analysisView = document.getElementById('analysis-view');
    const pageTitle = document.getElementById('page-title');
    const form = document.getElementById('run-form');
    const status = document.getElementById('run-status');
    const submit = document.getElementById('run-submit');
    const results = document.getElementById('run-results');
    if (!form) return;

    // ---- view switching ----
    function showView(which) {
        const analysis = which === 'analysis';
        analysisView.classList.toggle('hidden', !analysis);
        wikiContent.classList.toggle('hidden', analysis);
        tabAnalysis.classList.toggle('active', analysis);
        tabWiki.classList.toggle('active', !analysis);
        if (analysis) pageTitle.innerText = 'FEA Workbench';
    }
    tabWiki.onclick = () => { showView('wiki'); pageTitle.innerText = 'Welcome'; };
    tabAnalysis.onclick = () => showView('analysis');
    window.addEventListener('copilot:show-wiki', () => showView('wiki'));

    // ---- dynamic form behaviour ----
    const b1Kind = document.getElementById('rp-b1-kind');
    const b2Kind = document.getElementById('rp-b2-kind');
    function syncBearingUI() {
        document.getElementById('rp-b1-ball').classList.toggle('hidden', b1Kind.value !== 'ball');
        document.getElementById('rp-b2-ball').classList.toggle('hidden', b2Kind.value !== 'ball');
        const anyJournal = b1Kind.value === 'journal' || b2Kind.value === 'journal';
        document.getElementById('rp-film').classList.toggle('dimmed', !anyJournal);
        document.querySelectorAll('#rp-film input').forEach(i => i.disabled = !anyJournal);
    }
    b1Kind.onchange = syncBearingUI;
    b2Kind.onchange = syncBearingUI;

    const posCheck = document.getElementById('rp-pos-custom');
    posCheck.onchange = () => {
        document.querySelectorAll('#rp-pos-fields input').forEach(i => i.disabled = !posCheck.checked);
    };

    const num = (id) => parseFloat(document.getElementById(id).value);

    function bearingSpec(prefix) {
        const kind = document.getElementById(`rp-${prefix}-kind`).value;
        const spec = { kind };
        if (kind === 'ball') {
            spec.ball = {
                kxx_n_m: num(`rp-${prefix}-kxx`) * 1e6,
                kyy_n_m: num(`rp-${prefix}-kyy`) * 1e6,
                cxx_ns_m: num(`rp-${prefix}-cxx`) * 1e3,
                cyy_ns_m: num(`rp-${prefix}-cyy`) * 1e3,
            };
        }
        return spec;
    }

    // ---- run ----
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const params = {
            shaft: {
                diameter_m: num('rp-shaft-d') / 1000,
                length_m: num('rp-shaft-l') / 1000,
                youngs_modulus_pa: num('rp-shaft-e') * 1e9,
                density_kg_m3: num('rp-shaft-rho'),
            },
            disk: {
                diameter_m: num('rp-disk-d') / 1000,
                length_m: num('rp-disk-l') / 1000,
                mass_kg: num('rp-disk-m'),
                unbalance_kg_m: num('rp-unbal') * 1e-6,   // g·mm -> kg·m
            },
            bearing1: bearingSpec('b1'),
            bearing2: bearingSpec('b2'),
            journal_film: {
                diameter_m: num('rp-film-d') / 1000,
                length_m: num('rp-film-l') / 1000,
                radial_clearance_m: num('rp-clear') / 1e6,
                viscosity_pa_s: num('rp-visc'),
            },
            speed: {
                start_rad_s: num('rp-wmin'),
                stop_rad_s: num('rp-wmax'),
                step_rad_s: num('rp-wstep'),
            },
        };
        if (posCheck.checked) {
            params.positions = {
                bearing1_m: num('rp-pos-b1') / 1000,
                disk_m: num('rp-pos-disk') / 1000,
                bearing2_m: num('rp-pos-b2') / 1000,
            };
        }

        submit.disabled = true;
        status.className = '';
        status.textContent = 'Running finite-element analysis…';
        results.classList.add('hidden');
        try {
            const resp = await fetch('/api/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                const detail = typeof err.detail === 'string' ? err.detail
                    : (err.detail || []).map(d => `${(d.loc || []).join('.')}: ${d.msg}`).join('; ');
                throw new Error(detail || resp.statusText);
            }
            const r = await resp.json();
            renderResults(r);
            status.className = 'ok';
            status.textContent = 'Done — report ingested into the wiki.';
            if (window.copilotUI) await window.copilotUI.loadPages();
        } catch (err) {
            status.className = 'err';
            status.textContent = `Run failed: ${err.message}`;
        } finally {
            submit.disabled = false;
        }
    });

    function renderResults(r) {
        const rows = r.critical_speeds_rad_s.map((c, i) => {
            const hz = c / (2 * Math.PI);
            return `<tr><td>${['1st','2nd','3rd','4th','5th'][i] || (i + 1) + 'th'}</td>
                    <td>${c.toFixed(1)}</td><td>${hz.toFixed(1)}</td><td>${(hz * 60).toFixed(0)}</td></tr>`;
        }).join('');
        const pageId = r.report_slug.split('/').pop();
        results.innerHTML = `
            <h3>Results <span class="run-cite">(run: ${pageId})</span></h3>
            <table>
                <tr><th>Critical speed</th><th>rad/s</th><th>Hz</th><th>RPM</th></tr>
                ${rows || '<tr><td colspan="4">No critical speeds found in the analysed range</td></tr>'}
            </table>
            <p>Bearing reactions: <strong>R1 = ${r.bearing_reactions_n[0].toFixed(2)} N</strong>,
               <strong>R2 = ${r.bearing_reactions_n[1].toFixed(2)} N</strong>
               &middot; ${r.speed_points} speed points</p>
            <button type="button" id="open-report"><i class="fas fa-file-lines"></i> Open report page (plots)</button>`;
        results.classList.remove('hidden');
        document.getElementById('open-report').onclick = () => {
            if (window.copilotUI) window.copilotUI.loadPage(r.report_slug.split('/').pop());
        };
    }
});
