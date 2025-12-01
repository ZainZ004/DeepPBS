Dependency Upgrade Analysis for DeepPBS
Outdated Library Versions and Latest Releases
Below is a summary of DeepPBS’s key Python dependencies, their current versions (as used in the project), and the latest stable versions available as of late 2025. We also note the potential impact or significance of upgrading each (e.g. major changes or compatibility issues):
Library	Current Version	Latest Version (2025)	Upgrade Impact / Notes
PyTorch (torch)	2.3.0 (CUDA 12.1)	2.9.1 (Nov 12, 2025)[1]
Major update. Many improvements in performance (torch.compile etc.) and some breaking changes (e.g. default of torch.load now safer)[2][3]. Required for RTX 5090 support (includes newer CUDA).
Torchvision	0.18.0	0.24.1 (Nov 12, 2025)[4]
Major update. Must upgrade in tandem with PyTorch (version coupling: e.g. torch 2.9 ↔ torchvision 0.24)[5]. New models and transforms; minimal API breakage expected.
Torchaudio	2.3.0	2.9.1 (Nov 12, 2025)[6]
Major update. Align with PyTorch version. Some features deprecated in 2.8 and removed in 2.9 (maintenance mode changes)[7].

PyTorch Geometric (PyG)	~2.5.0 (tested)[8]
2.7.0 (Oct 15, 2025)[9]
Moderate update. Backwards-compatible for most GNN APIs, but update extension packages (scatter/sparse/cluster) to match new PyTorch. New utilities (e.g. torch.compile support)[10].

– torch_scatter	(for torch 2.3)	2.1.2 (latest)[11]
Rebuild for new torch/CUDA. Minor internal changes (performance fixes, NaN handling)[12].

– torch_sparse	(for torch 2.3)	0.6.18 (latest)[13]
Rebuild for new torch. No API change; ensure METIS optional dep if used.
– torch_cluster	(for torch 2.3)	1.6.3 (latest)[14]
Rebuild for new torch. No API change.
Biopython	1.83	1.86 (Oct 28, 2025)[15]
Minor update. Mostly bug fixes and new file format support. Low risk; ensure tests pass (no major API changes up to 1.86).
Logomaker	0.8.x (unpinned)	0.8.7 (Mar 25, 2025)[16]
Minor update. Likely at 0.8.5/0.8.6 currently; updating to 0.8.7 fixes known issues (e.g. caching)[17]. No breaking changes expected (same major version).
Matplotlib	3.5.2	3.10.7 (Oct 9, 2025)[18]
Major update (several minor releases). New features (3.6–3.10) and some API tweaks. Mostly backward-compatible; check for deprecations (e.g. plt.rcParams keys or colormap defaults) in plots.
NetworkX	Unpinned (likely 3.0)	3.6 (Nov 24, 2025)[19]
Major update if <3.0 before. NetworkX 3.x dropped some deprecated 2.x APIs. If project was on 2.x, migrating to 3.6 requires adjusting any removed aliases (e.g. Graph methods returning views) and ensuring Python ≥3.11[20].

Pandas	1.4.4	2.3.3 (Sep 29, 2025)[21]
Major update (2.x). Pandas 2.0+ introduces the Apache Arrow backend by default and removed older pandas APIs. Potential adjustments needed for CSV/JSON IO (Arrow data types) and any deprecated 1.x methods.
PDB2PQR	3.x (unversioned)	3.7.1 (Dec 28, 2024)[22]
Moderate update. Newer versions require Python ≥3.11[23]. CLI interface (pdb2pqr) remains similar. Check if default forcefields or flags changed in 3.7.1.
SciPy	1.14.1	1.16.3 (Oct 28, 2025)[24]
Minor/Moderate update. Incremental releases with new functions and deprecations. Ensure no use of any function removed by 1.16 (unlikely). Requires NumPy upgrade if needed (likely already up-to-date via PyTorch).
Seaborn	0.13.2	0.13.2 (Latest)[25]
Up-to-date. No newer stable release beyond 0.13.2 as of 2025. No action needed (but test with new Matplotlib/Pandas).
FreeSASA	2.2.1	2.2.1 (Latest)[26]
Up-to-date. 2.2.1 is latest stable. No update needed, but rebuild C library if Python version changes.
Table: DeepPBS Python dependencies – current vs latest versions and upgrade impact.
Recommended Upgrade Actions and Code Adjustments
Upgrading the above libraries will bring performance improvements and compatibility with new hardware, but may require code or configuration changes. Below we outline recommendations per library (or logical group of libraries), including any needed code modifications and configuration updates:
•	PyTorch and GPU Ecosystem (torch, torchvision, torchaudio, PyG): Upgrade PyTorch to 2.9.1 (with a CUDA toolkit supporting RTX 5090, e.g. CUDA 12.7+). Because these libraries are tightly coupled, install matching versions: e.g. PyTorch 2.9.1 with torchvision 0.24.1 and torchaudio 2.9.1[5]. Also upgrade PyTorch Geometric to the latest (2.7.0) and reinstall its companion packages (torch_scatter 2.1.2, torch_sparse 0.6.18, torch_cluster 1.6.3) compiled against the new PyTorch/CUDA. This ensures GPU kernels are optimized for the new CUDA version. After upgrading, re-run the installation commands (possibly adjusted to a new PyTorch wheel URL) as in the README, but pointing to the torch 2.9 wheels (and the corresponding PyG wheels for torch 2.9 + CUDA 12.x)[27].
Code changes: Little to no changes in model code are expected, as PyTorch’s API from 2.3 to 2.9 remains largely backward compatible. However, watch for deprecation warnings and adjust accordingly: - The default behavior of torch.load now only loads state_dict weights by default for security[2]. If DeepPBS uses torch.load to load full model objects, set weights_only=False or switch to loading state_dicts explicitly.
- If any custom CUDA extension code is present (unlikely in DeepPBS), it may need recompilation with C++17 and CXX11 ABI to match PyTorch 2.6+ builds[28]. - PyTorch 2.9 expands support for newer GPUs and drops support for very old GPUs (sm_50–sm_60)[29], so ensure the NVIDIA driver is updated to the latest version that supports the RTX 5090 architecture. - No changes are needed for high-level PyTorch APIs (autograd, nn.Module, etc.) – these remain compatible. But after upgrade, run the training and inference to confirm that results are numerically similar (some GPU kernels or random seed behavior might differ slightly).
•	TorchVision/TorchAudio Models: If DeepPBS uses any pretrained vision models or audio features (likely not, since it’s a bioinformatics project), confirm that their APIs haven’t changed. For example, torchvision 0.24 introduced new transforms and datasets, but existing functions remain. No code changes anticipated beyond updating import versions.
•	PyTorch Geometric and Graph Libraries: DeepPBS relies on geometric deep learning (PyG for graph neural networks). After upgrading PyG to 2.7, verify that the custom dataset and model classes still work:
•	Data and transforms: PyG 2.x is backwards compatible with 2.0. If the project built graphs with torch_geometric.data.Data or used PyG transforms, these should function the same. Confirm that the version of NetworkX used in any graph conversion is compatible (PyG’s to_networkx or similar functions might expect NetworkX 2.x or 3.x – PyG 2.7 supports NetworkX 3.x).
•	Scatter/Sparse Ops: The torch_scatter, torch_sparse, torch_cluster packages must be updated as noted. No code changes needed for their usage (APIs unchanged), but these must match the new torch version or you’ll get runtime import errors. Use the official PyG wheel index for torch 2.9 (CUDA 12.x) as done previously (e.g. pip install torch_scatter -f https://data.pyg.org/whl/torch-2.9.0+cu127.html).
•	Potential API differences: Check the PyG changelog for any renamed classes or arguments since 2.5. For instance, ensure any usage of pyg.loader or neighbor samplers still works (should be fine). The project’s training scripts (e.g. submit_cross.sh and model definition in deeppbs/nn/layers/*.py) likely remain valid.
•	Scientific Computing Stack (NumPy/SciPy/Matplotlib/Pandas): These libraries underpin data processing and plotting. Upgrading them improves performance and compatibility:
•	NumPy: (Indirectly upgraded via PyTorch’s requirements). Ensure the installed NumPy is at least the version required by new SciPy. PyTorch 2.9 wheels typically come with a recent NumPy.
•	SciPy (1.16.x): No code changes expected. SciPy’s API changes are minimal between 1.14 and 1.16 (mostly additions). If DeepPBS uses SciPy (perhaps for statistics or clustering), verify those functions still behave the same. Run any unit tests for functions like scipy.stats or scipy.spatial that are used.
•	Matplotlib (3.10.x): After upgrading, test all plotting functions. Matplotlib had some minor API updates (e.g., 3.6 changed how figure tight layout is handled, 3.7/3.8 added new colormaps). Specifically:
o	If DeepPBS customizes plot aesthetics, check that style parameters (rcParams keys) remain valid. E.g., some 3.5 rcParams were renamed by 3.8.
o	If any warnings appear (Matplotlib often prints deprecation warnings for deprecated function arguments), update the code accordingly. For example, plt.legend() parameters or AxesGrid usage might warn if syntax changed.
o	Seaborn 0.13.2 remains the same, but behind the scenes it will use the new Matplotlib and Pandas. Run the logo plotting (if DeepPBS uses Logomaker/Seaborn for sequence logos) to ensure the visuals are as expected.
•	Pandas (2.3.x): This is a major jump from 1.4. Key considerations:
o	Arrow dtype: Pandas 2.x may default to using Arrow for certain data types (especially if writing/reading Parquet or CSV). If DeepPBS only uses pandas for simple DataFrame operations (e.g. reading a CSV of input sequences or logging results), it should mostly work as before. But if code explicitly relies on pandas internal types (e.g. comparing dtypes names, or using pd.DataFrame.to_numpy() which might return Arrow-backed arrays), test these paths. You can force pandas to use the “legacy” numpy backend via environment variable if needed, but it’s better to adapt to Arrow going forward[30].
o	Check for any usage of methods removed in Pandas 2.0. For example, DataFrame.append() was deprecated and removed – if the code used it, switch to pd.concat. Also, Panel (3D data structure) was long removed (probably not used here).
o	Pandas and Seaborn: After upgrade, Seaborn might produce warnings due to changes in pandas indexing. For instance, seaborn plotting functions might warn about pandas categorical handling changes (make sure to update seaborn if a new version appears, but currently 0.13.2 is latest and supports pandas 2.x).
o	In summary, run any data loading and analysis portion of DeepPBS to catch errors. The likely needed code tweaks are small (e.g., replace deprecated calls, adjust for stricter type handling).
•	Bioinformatics Libraries (Biopython, PDB2PQR, FreeSASA): These handle domain-specific logic (parsing PDB files, adding charges, surface area calcs):
•	Biopython 1.86: Should work out-of-the-box. Biopython rarely breaks backward compatibility; the Bio.PDB module (used for parsing structures in DeepPBS) has only seen improvements. One thing to note: ensure the new Biopython still supports reading mmCIF if needed (it does). If DeepPBS relied on any private Biopython API, verify it’s still present.
•	PDB2PQR 3.7.1: The CLI command pdb2pqr is still the entry point (with flags like --ff=AMBER as used in DeepPBS[31]). Upgrade this via pip. Important: PDB2PQR 3.7+ requires Python 3.11+[23]. This means you should upgrade the base Python in your environment (DeepPBS currently used Python 3.10 in Conda[32]). Plan to move to Python 3.11 or 3.12 to accommodate PDB2PQR and NetworkX upgrades. After installing the new PDB2PQR, test the runPDB2PQR() function in DeepPBS:
o	The subprocess call to pdb2pqr with --ff=AMBER --keep-chain should still produce a .pqr file[31]. Check PDB2PQR release notes to confirm if AMBER forcefield is still supported or renamed (likely yes, AMBER is standard).
o	If any errors occur (for example, PDB2PQR might have changed how it handles certain PDB issues), you may need to adjust input or catch new error messages. The DeepPBS code already handles a common issue by padding coordinates in PQR files[33][34] – ensure this hack is still needed or works (it should).
o	Also verify that PDB2PQR’s output parsing (extracting charge and radius from columns) still aligns with the file format. The code uses fixed column indices[35], which should remain correct as PQR format hasn’t changed.
•	FreeSASA 2.2.1: No new version, but if upgrading Python, reinstall FreeSASA’s Python binding (it provides a C library wrapper). FreeSASA’s API (freesasa.calc() etc.) remains the same. Just ensure it’s invoked with correct parameters if Python version changed (no code change anticipated). Because FreeSASA is used via its Python module in DeepPBS (likely to calculate solvent accessible surface area for certain atoms), run that part to confirm it still returns expected values. No known breaking changes since 2.2.1 is the latest.
•	NetworkX (Graph utility): The environment didn’t pin NetworkX, so it likely installed the then-latest. By 2025 that’s NetworkX 3.x. DeepPBS might use NetworkX for ancillary tasks (perhaps analyzing graph structure or converting networks). If the project was using NetworkX 2.x and we now have 3.6:
•	Notable change: NetworkX 3+ requires Python 3.11[20] and removed some deprecated methods. For example, in NetworkX 2, one could iterate over G.nodes() to get a list – in 3.x it’s a view, but iteration still works (just cast to list if needed). Many methods like G.neighbors etc. remain, so most simple uses won’t break.
•	Action: Run any code that invokes NetworkX (perhaps in constructing graph inputs to PyG or evaluating graph connectivity). If an error arises (AttributeError for a removed function), consult the NetworkX migration notes. Typical fixes involve using list(G.nodes) instead of G.nodes() if the code expected a list, or updating any algorithm usage that changed (unlikely).
•	Given DeepPBS is primarily using PyG for graph computations, NetworkX might be a minor part (maybe for visualization or reading .graphml if any). So this upgrade is low-risk with minimal code changes.
•	Visualization and Logo Tools (Seaborn, Logomaker): Seaborn is already at latest (0.13.2) and supports the newer matplotlib and pandas. Logomaker 0.8.7 is a minor bump:
•	After upgrading, generate a test sequence logo (DeepPBS likely uses Logomaker to visualize position weight matrices of DNA specificity). Ensure that the logos look correct. Logomaker’s API (e.g. Logo(dataframe)) hasn’t changed, but behind the scenes it uses matplotlib. No code change expected, just verify that text and colors render the same with Matplotlib 3.10.
•	If any warnings arise (for example, if Logomaker used any matplotlib function that changed), consider checking the Logomaker repo for an update or applying a small patch. As of 0.8.7, multiple bugfixes were included[17] which presumably make it compatible with latest matplotlib.
•	Conda Environment & Installation: Update the Conda environment YAML (deeppbs_linux.yml) with new version pins:
•	Set python=3.11 (or 3.12) environment, since several packages now require it.
•	Update pytorch, torchvision, torchaudio versions (if using conda install). Note that PyTorch no longer publishes to the main Conda channel as of 2.6[36], so you may either use pip in the conda env or rely on Conda-Forge. The safest route is to use pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 -c pyTorch -c nvidia or similar pip commands as shown in the README, adjusting versions.
•	Adjust the pip installations: use pip install biopython==1.86 pandas==2.3.3 matplotlib==3.10.7 scipy==1.16.3 seaborn==0.13.2 freesasa==2.2.1 logomaker==0.8.7 (networkx and PDB2PQR will get pulled in as dependencies or install them explicitly: networkx==3.6 pdb2pqr==3.7.1). The updated environment file will document these changes for future reproducibility.
•	General Code Audit for Deprecations: After upgrading all, run DeepPBS’s scripts with a deprecation warning filter turned on (e.g. -Wd flag or using Python’s warnings module). This will show if any functions are slated for removal. Address any such warnings proactively:
•	For example, if Biopython warns that a certain parser is deprecated, switch to the recommended one. Or if PyTorch warns about an old alias (e.g. using torch.tensor on a list of floats without dtype might warn after NumPy upgrade), fix those to avoid trouble later.
In summary, most code changes are expected to be minor, given the upgrades are within the same major branch (except pandas). The primary adjustments are updating the environment, ensuring compatibility with Python 3.11+, and modifying a few lines if older pandas or networkx usage was present. Each upgraded library should be tested in isolation first (to pinpoint any issues), then as an integrated pipeline.
Post-Upgrade Testing Plan (Unit and Integration Tests)
After performing the upgrades, it’s crucial to verify that DeepPBS’s core functionality remains correct. We propose a suite of tests to validate each component of the software. These tests will help catch any subtle issues introduced by the library changes:
•	1. Data Preprocessing Tests: Ensure that raw input structures (PDB/CIF files) are processed into the model-ready format correctly.
•	PDB→PQR Conversion: Provide a small sample PDB file (e.g., a single protein-DNA complex from the project’s dataset) to the runPDB2PQR function and verify it produces a .pqr file and a StructureData object without errors. Assert that the output PQR contains expected entries (ATOM lines with charges and radii). For example, after running, parse the PQR and check that no atom has a zero radius (DeepPBS sets a minimum 0.6 for any missing radii[37]). This ensures PDB2PQR and Biopython integration still works.
•	Feature NPZ Generation: If DeepPBS has a function to convert PQR + 3DNA outputs into feature arrays (perhaps in run/process_and_predict.sh), simulate a run on a known PDB. Then load the resulting NPZ and verify dimensions and basic statistics (e.g., expected number of nodes equals heavy atoms count, no NaNs in features). This validates that upstream dependencies (FreeSASA, DSSP via Biopython, etc.) produce consistent results.
•	Hydrogen Stripping: Test the stripHydrogens utility on a PDB containing hydrogens to ensure it still removes all H atoms via Biopython calls. This can be done by counting atoms before and after stripping.
•	2. Model Forward-Pass Tests: Confirm the neural network components operate correctly on sample data.
•	Graph Construction: Construct a minimal torch_geometric.data.Data graph manually (e.g., a toy graph with a few nodes and edges, or take a small real processed example). Pass it through the DeepPBS model’s forward method. Check that the output tensor has the expected shape and range. For instance, if the model predicts a Position Weight Matrix or binding specificity scores, ensure the output dimensions match the alphabet size and that probabilities sum to 1 (if applicable).
•	Deterministic Output Check: Use a fixed random seed and run the model on the sample input before and after upgrades. The outputs may not match exactly numerically due to changed random initializations or kernel algorithms, but the test can at least ensure the model runs without runtime errors (e.g., no dimension mismatches, no attribute errors) and yields reasonable values (e.g., within expected bounds). If the model had any unstable operations that new library versions handle differently (for example, PyTorch’s dropout or batchnorm updates), consider loosening the test to just verify consistency in shapes and the presence of expected structural patterns in output rather than exact values.
•	3. Training Routine Tests: Though full training is lengthy, we can simulate a short training loop to ensure nothing breaks:
•	One-Batch Training: Initialize the model and run a single training step on a small batch. Use dummy data: e.g., create 1–2 graph samples with random features of the correct size. Perform a forward and backward pass. Assert that:
o	The loss decreases or at least computes without issue.
o	Model parameters get updated (check that parameters before and after optimizer step are not identical).
o	No exceptions are raised during backward (this catches issues like missing gradient definitions or GPU memory errors on 5090).
•	This test will quickly flag any incompatibilities in the training code (for example, if an optimizer’s API changed or if any layer is not handling new data types well).
•	4. Output Consistency Tests: Validate that the end-to-end pipeline produces outputs in the expected format:
•	Prediction Output: Run the process_and_predict.sh pipeline on a known input (perhaps provided in the project’s examples, e.g., PDB id 5x6g as mentioned in the README). Compare the new output to a baseline (if available) or at least check the presence and structure of results:
o	Confirm that a results directory is created with predict/ subfolder containing a Position Weight Matrix file. If a baseline PWM from before the upgrade is known, compare the shape and perhaps certain characteristics (e.g., consensus sequence or motif length). Minor numerical differences are expected, but overall motif should not be entirely different.
o	If the pipeline generates an interpretation/ folder (when run with the -m flag), ensure it contains the PyMOL session or residue-wise scores. You can test that the vis_interpret.py script runs without error and produces an output .pse file. Since automating PyMOL might be hard in unit tests, you can instead have the script output numeric scores (it likely prints or saves per-atom importance scores). Validate that these scores are reasonable (e.g., not all zero, and specific to heavy atoms).
•	Visualization: For the sequence logo generation (if part of pipeline), run the Logomaker plotting code with a known PWM (for example, a simple PWM with a known motif). Have the code generate a matplotlib figure (you can use a non-GUI backend for testing). Assert that no exceptions occur during plotting. Optionally, verify that the axes labels and title match expected strings (since those might be set by the code). This ensures that the combination of Matplotlib + Seaborn + Logomaker still works in tandem.
•	5. Performance & Memory Check on RTX 5090: While not a traditional unit test assertion, it is wise to include a simple GPU compatibility test:
•	Allocate a tensor on the RTX 5090 (e.g., torch.rand(1000,1000, device='cuda')) and perform a sample operation (matrix multiply) to ensure PyTorch recognizes the GPU and runs computations. This is more of an integration test to catch any CUDA initialization issues. It should pass if PyTorch and CUDA drivers are correctly set up.
•	During a full run (processing + prediction on a sample), monitor memory usage. The new GPU has a different memory profile; ensure that there are no obvious memory leaks by checking that GPU memory is freed after model inference (PyTorch should handle this, but a test that calls the pipeline twice in a row can ensure the second run doesn’t OOM, indicating proper cleanup).
•	6. Regression Tests on Known Results: If the team has previous benchmark results (e.g., performance metrics or specific predictions from a paper or earlier version), use them for regression testing:
•	For example, if DeepPBS was evaluated on a benchmark set and achieved certain accuracy or AUC, run the upgraded version on a subset of that data and ensure the metrics are in the same ballpark. Significant drops in performance would indicate a potential issue introduced by the upgrade.
•	If exact matching of results is unrealistic (due to randomness), focus on qualitative consistency: the model should still preferentially predict correct binding specificities. For instance, if a particular protein-DNA pair was known to have a strong motif in the original version, confirm that the motif is still recovered in the upgraded version’s output.
By implementing the above tests, we cover the major functions: preprocessing, model inference, training, and post-processing/visualization. Automating these tests (using a framework like pytest) and running them after the upgrade will quickly highlight any divergence. This strategy provides confidence that upgrading dependencies has not broken DeepPBS’s functionality.
NVIDIA RTX 5090 Compatibility Notes
The RTX 5090 represents a new generation of NVIDIA GPUs, and ensuring DeepPBS runs efficiently on it requires attention to CUDA and driver compatibility:
•	CUDA Toolkit and Drivers: Upgrade the NVIDIA driver to the latest version that supports the RTX 5090 architecture. New GPUs often come with a new compute capability (e.g., Ada Lovelace (RTX 40 series) was compute 8.9; the RTX 5090 might be even higher). Using an up-to-date CUDA 12.x (or CUDA 13, if released) is crucial. PyTorch 2.9.1 ships with support for recent CUDA versions (the PyTorch 2.6 release, for example, used CUDA 12.6.3 for its binaries[38]). Ensure that the PyTorch binary installed includes support for the 5090’s compute capability – official wheels typically include PTX for future architectures, so the 2.9 wheel should work out of the box on 5090. In summary, choose a PyTorch build with CUDA ≥12.6. For instance, installing pytorch-cuda=12.8 (if using Conda) or the pip wheel built against CUDA 12.8 would be appropriate.
•	Recompile Custom Ops for 5090 (if needed): PyTorch Geometric’s extension packages (scatter/sparse/cluster) should be installed in versions matching the new CUDA. These wheels are often built for specific CUDA versions (e.g., cu121, cu127). Use the correct URL (as noted earlier) to get wheels that include sm binaries for the latest GPUs. If pre-built wheels don’t yet list support for the 5090, they may still function via PTX JIT compilation, but for optimal performance you might choose to compile them from source with TORCH_CUDA_ARCH_LIST set to include the 5090’s compute capability. For example, if RTX 5090 is compute capability 9.0, set TORCH_CUDA_ARCH_LIST="9.0" before pip installing PyG extensions to generate optimized cubins[39]. This will ensure those custom CUDA kernels run at full speed on the new hardware.
•	Check for Dropped Legacy Support: As noted, newer PyTorch versions dropped support for very old GPUs (sm_50–sm_60) to keep binary size down[29]. This doesn’t affect the 5090 directly (it’s a forward-looking change), but it highlights that the focus is on newer architectures. The 5090 will be fully supported provided you’re on the latest PyTorch/torchvision stack.
•	Utilizing RTX 5090 Features: The RTX 5090 may offer improved tensor core performance or larger memory. PyTorch 2.9 likely already enables Tensor Cores for FP16/BF16 by default on Ampere/Ada GPUs. You should ensure mixed precision training is leveraged if appropriate (PyTorch’s torch.cuda.amp can significantly speed up inference/training on newer GPUs). This isn’t a dependency upgrade per se, but a configuration tweak: e.g., using with torch.autocast(device_type='cuda', dtype=torch.float16): around inference could yield speed-ups on 5090. This should be tested for numerical stability in DeepPBS’s case, but it’s worth noting as a hardware-specific optimization.
•	Memory and Batch Size: The RTX 5090 likely has more VRAM. After upgrading, you might increase batch sizes or parallelism in DeepPBS. This isn’t required for compatibility, but to use the hardware fully. It would be an additional step: once the software is verified to work, experiment with larger batch processing to exploit the 5090’s capacity (ensuring the code does not have any hidden assumptions about batch size or overflow).
•	Driver Compatibility with PyTorch: Make sure the NVIDIA driver version is compatible with the CUDA runtime version used by PyTorch. You can verify this by running a simple CUDA operation (as mentioned in tests) or using torch.cuda.is_available() and torch.cuda.nccl.version(). If there is a mismatch, you may need to update the driver. NVIDIA’s documentation usually states the minimum driver version for a given CUDA (for example, CUDA 12.8 might require driver ≥ 535.xx – check NVIDIA’s CUDA release notes for exact numbers).
In essence, running DeepPBS on an RTX 5090 should not require code changes in DeepPBS itself – it’s about using updated libraries that recognize the GPU. By moving to the latest PyTorch and CUDA, we ensure the software can see and utilize the card. The steps are: 1. Upgrade PyTorch + CUDA as described. 2. Verify GPU is recognized (torch.cuda.device_count() should include the 5090). 3. Run end-to-end on the GPU to ensure no CUDA runtime errors. This includes the PyG ops, which, after proper installation, should run fine. If any issue arises (e.g., an undefined symbol or illegal instruction in a CUDA kernel), it’s likely due to a mismatched binary – reinstalling the correct wheel or building from source with the proper arch flag will fix it.
Lastly, document these environment changes for the team. It’s helpful to note in the README (or internal docs) that the pipeline was tested on NVIDIA RTX 5090 with CUDA X.Y and Driver Z.ZZ. This gives users confidence that the upgraded DeepPBS will run on new GPU machines.
Sources:
•	DeepPBS installation guide (original versions)[40][41]
•	PyTorch, TorchVision, Torchaudio latest releases[1][4][6]; PyTorch release notes[3][38]
•	PyTorch Geometric and extensions versions[9][11][13][14]
•	Biopython and PDB2PQR version info[15][22][23]
•	Scientific libraries (Pandas, SciPy, NetworkX, Matplotlib) latest versions[21][24][19][18], and notes on Pandas 2.0 arrow integration[30].
•	Excerpts from DeepPBS code showing usage of PDB2PQR and other components[31][35].
•	Torch Scatter docs on setting architecture flags[39] and PyTorch dropping old sm support[29].
________________________________________
[1] torch · PyPI
https://pypi.org/project/torch/
[2] [3] [28] [29] [36] [38] Releases · pytorch/pytorch · GitHub
https://github.com/pytorch/pytorch/releases
[4] [5] torchvision · PyPI
https://pypi.org/project/torchvision/
[6] [7] torchaudio · PyPI
https://pypi.org/project/torchaudio/
[8] [27] [32] [40] [41] GitHub - timkartar/DeepPBS: Geometric deep learning of protein–DNA binding specificity
https://github.com/timkartar/DeepPBS
[9] [10] torch-geometric · PyPI
https://pypi.org/project/torch-geometric/
[11] torch-scatter: Complete Python Package Guide & Tutorial [2025]
https://generalistprogrammer.com/tutorials/torch-scatter-python-package-guide
[12] Releases · rusty1s/pytorch_scatter - GitHub
https://github.com/rusty1s/pytorch_scatter/releases
[13] torch-sparse · PyPI
https://pypi.org/project/torch-sparse/
[14] torch-cluster · PyPI
https://pypi.org/project/torch-cluster/
[15] Biopython · Biopython
https://biopython.org/
[16] logomaker - PyPI
https://pypi.org/project/logomaker/
[17] Releases · jbkinney/logomaker - GitHub
https://github.com/jbkinney/logomaker/releases
[18] matplotlib · PyPI
https://pypi.org/project/matplotlib/
[19] [20] networkx · PyPI
https://pypi.org/project/networkx/
[21] pandas · PyPI
https://pypi.org/project/pandas/
[22] [23] pdb2pqr · PyPI
https://pypi.org/project/pdb2pqr/
[24] SciPy
https://scipy.org/
[25] seaborn · PyPI
https://pypi.org/project/seaborn/
[26] freesasa · PyPI
https://pypi.org/project/freesasa/
[30] What's new in 2.3.3 (September 29, 2025) - Pandas
https://pandas.pydata.org/docs/whatsnew/v2.3.3.html
[31] [33] [34] [35] [37] run_pdb2pqr.py
https://github.com/timkartar/DeepPBS/blob/8bfb211dd67f02877841f6f33aa493ddf7daedf9/deeppbs/run_pdb2pqr.py
[39] torch-scatter · PyPI
https://pypi.org/project/torch-scatter/
