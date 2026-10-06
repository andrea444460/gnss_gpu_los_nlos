import re

def patch_html(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 1. Patch Edifici OSM (trasparenza al 40%)
    html_content = re.sub(
        r'(viewer\.scene\.primitives\.add\(tileset\);)',
        r'tileset.style = new Cesium.Cesium3DTileStyle({ color: "color(\'white\', 0.4)" });\n      \1',
        html_content
    )

    # 2. Patch CSS per rendere il pannello C/N0 ancorato in basso, non bloccante e con sfondo sfocato
    html_content = re.sub(
        r'#cnrBackdrop\s*\{[^}]+\}',
        """#cnrBackdrop {
    display: none;
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 50;
    pointer-events: none; /* Lascia passare i clic sulla mappa retrostante */
  }
  #cnrBackdrop.show { display: block; }
  #cnrPanel { 
    pointer-events: auto; 
    background: rgba(18, 23, 34, 0.95); 
    backdrop-filter: blur(4px); 
  }""",
        html_content
    )

    # 3. Patch HTML: Nasconde la "select multiple" e inserisce un container per la lista a checkbox
    html_content = re.sub(
        r'(<select id="selSatellite".*?>\s*</select>)',
        r'<div id="satelliteCheckboxList" style="max-height: 120px; overflow-y: auto; background: #1a1d26; border: 1px solid #445; border-radius: 6px; padding: 6px; display: flex; flex-direction: column; gap: 4px; width: 100%; font-family: monospace;"></div>\n        <select id="selSatellite" multiple style="display:none;"></select>',
        html_content
    )

    # 4. Iniezione dello Script UI a fine pagina per raggruppare i controlli in Tab e gestire le Checkbox
    ui_script = """
<script>
// --- PATCH INTERFACCIA UTENTE (AUTO-GENERATA) ---
(function() {
  // Sincronizzazione automatica bidirezionale tra la select originale (ora nascosta) e le nuove checkbox
  const selSat = document.getElementById('selSatellite');
  const chkList = document.getElementById('satelliteCheckboxList');
  if(selSat && chkList) {
      function updateCheckboxes() {
        chkList.innerHTML = '';
        Array.from(selSat.options).forEach(opt => {
          const lbl = document.createElement('label');
          lbl.style.cssText = 'font-size: 11px; color: #ccc; display: flex; align-items: center; gap: 6px; cursor: pointer; padding: 2px 4px; border-radius: 4px;';
          lbl.onmouseover = () => lbl.style.background = '#2a3142';
          lbl.onmouseout = () => lbl.style.background = 'transparent';
          
          const chk = document.createElement('input');
          chk.type = 'checkbox';
          chk.value = opt.value;
          chk.checked = opt.selected;
          
          // Quando si clicca una checkbox, aggiorniamo il menu a tendina nascosto e simuliamo il 'change'
          chk.addEventListener('change', () => {
            opt.selected = chk.checked;
            selSat.dispatchEvent(new Event('change'));
          });
          lbl.appendChild(chk);
          lbl.appendChild(document.createTextNode(' ' + opt.text));
          chkList.appendChild(lbl);
        });
        if(selSat.options.length === 0) {
            chkList.innerHTML = '<span style="color:#666; font-size:11px;">Nessun satellite disponibile</span>';
        }
      }
      
      // Il MutationObserver intercetta quando il tuo codice preesistente popola i satelliti
      const observer = new MutationObserver(updateCheckboxes);
      observer.observe(selSat, { childList: true });
      updateCheckboxes();
  }

  // Costruzione delle Schede (Tabs) nel Pannello Overlay
  const playbackDiv = document.getElementById('playback');
  if(playbackDiv) {
      const style = document.createElement('style');
      style.textContent = `
        .tab-header { display: flex; border-bottom: 1px solid #445; margin-bottom: 12px; padding-bottom: 4px; }
        .tab-btn { flex: 1; background: transparent; border: none; color: #889; padding: 6px 0; cursor: pointer; font-size: 12px; border-bottom: 2px solid transparent; transition: 0.2s; }
        .tab-btn:hover { color: #fff; }
        .tab-btn.active { color: #3b7ddd; border-bottom: 2px solid #3b7ddd; font-weight: bold;}
        .tab-content { display: none; }
        .tab-content.active { display: block; }
      `;
      document.head.appendChild(style);

      const tabHeader = document.createElement('div');
      tabHeader.className = 'tab-header';
      tabHeader.innerHTML = `
        <button class="tab-btn active" id="btnTabPlay">Riproduzione</button>
        <button class="tab-btn" id="btnTabViz">3D & Opzioni</button>
        <button class="tab-btn" id="btnTabFilt">Filtri</button>
      `;
      
      const tabPlay = document.createElement('div'); tabPlay.className = 'tab-content active';
      const tabViz = document.createElement('div'); tabViz.className = 'tab-content';
      const tabFilt = document.createElement('div'); tabFilt.className = 'tab-content';

      const rows = Array.from(playbackDiv.querySelectorAll('.row'));
      if(rows.length >= 4) {
          // Ricollocazione dei controlli (preserva tutti gli EventListener preesistenti)
          tabPlay.appendChild(rows[0]); // Pulsanti Playback
          tabPlay.appendChild(rows[2]); // Slider Epoch
          
          const optionsWrapper = document.createElement('div');
          optionsWrapper.className = 'row';
          optionsWrapper.style.flexDirection = 'column';
          optionsWrapper.style.alignItems = 'flex-start';
          optionsWrapper.style.gap = '8px';
          
          // Estrai Follow RX e Multipath dalla prima riga
          const chkFollow = document.getElementById('chkFollowRx');
          if(chkFollow) optionsWrapper.appendChild(chkFollow.parentNode);
          const chkMulti = document.getElementById('chkShowMultipath');
          if(chkMulti) optionsWrapper.appendChild(chkMulti.parentNode);
          optionsWrapper.appendChild(rows[1]); // OSM Toggle
          
          tabViz.appendChild(optionsWrapper);
          tabFilt.appendChild(rows[3]); // Filtri satellitari
      }
      
      const hint = document.getElementById('playbackHint');
      playbackDiv.innerHTML = ''; 
      playbackDiv.appendChild(tabHeader);
      playbackDiv.appendChild(tabPlay);
      playbackDiv.appendChild(tabViz);
      playbackDiv.appendChild(tabFilt);
      if(hint) playbackDiv.appendChild(hint);

      document.getElementById('btnTabPlay').onclick = (e) => switchTab(e.target, tabPlay);
      document.getElementById('btnTabViz').onclick = (e) => switchTab(e.target, tabViz);
      document.getElementById('btnTabFilt').onclick = (e) => switchTab(e.target, tabFilt);
      
      function switchTab(btn, content) {
          document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
          btn.classList.add('active');
          content.classList.add('active');
      }
  }
})();
</script>
</body>"""
    
    html_content = html_content.replace('</body>', ui_script)

    # Salva il nuovo file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ Interfaccia aggiornata con successo! Salvato come {output_file}")

if __name__ == "__main__":
    patch_html("C:/Users/Andrea/Desktop/KTL_Label_geometrico/viz/0610_KLT1_203_viz.html", "C:/Users/Andrea/Desktop/KTL_Label_geometrico/viz/0610_KLT1_203_viz_patched.html")