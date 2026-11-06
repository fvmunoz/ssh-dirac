import sys
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ==========================================
# 1. Physics Classes
# ==========================================
class BaseLattice1D:
    def __init__(self, N, pbc=False):
        self.N = N
        self.pbc = pbc
        self.dim = 2 * N
        self.H = np.zeros((self.dim, self.dim), dtype=complex)
        self.evals, self.evecs, self.ipr, self.sublattice_pol = None, None, None, None

    def add_hopping(self, i, j, amplitude):
        if not self.pbc and (i >= self.dim or j >= self.dim or i < 0 or j < 0):
            return
        i, j = i % self.dim, j % self.dim
        if i == j:
            self.H[i, i] += amplitude
        else:
            self.H[i, j] += amplitude
            self.H[j, i] += np.conj(amplitude)

    def solve(self):
        self.evals, evecs_cols = np.linalg.eigh(self.H)
        self.evecs = evecs_cols.T 
        self.ipr = np.sum(np.abs(self.evecs)**4, axis=1)
        
        prob_A = np.abs(self.evecs[:, 0::2])**2
        prob_B = np.abs(self.evecs[:, 1::2])**2
        numerator = np.sum(np.abs(prob_A - prob_B), axis=1)
        denominator = np.sum(prob_A + prob_B, axis=1)
        self.sublattice_pol = numerator / np.where(denominator == 0, 1, denominator)
        
        return self.evals, self.evecs

class SSHLatticeModel(BaseLattice1D):
    def __init__(self, N, pbc=False, v=1.0, w=0.5, tA=0.0, tB=0.0, u1=0.0, u2=0.0, delta=0.0, dw_hopping=False, dw_staggering=False):
        super().__init__(N, pbc)
        self.v_arr = np.full(N, v, dtype=float)
        self.w_arr = np.full(N, w, dtype=float)
        self.delta_arr = np.full(N, delta, dtype=float)
        
        mid = N // 2
        if dw_hopping:
            self.v_arr[mid:], self.w_arr[mid:] = w, v
        if dw_staggering:
            self.delta_arr[mid:] = -delta
            
        self.tA, self.tB, self.u1, self.u2 = tA, tB, u1, u2
        self._build_matrix()

    def _build_matrix(self):
        for n in range(self.N):
            iA, iB = 2 * n, 2 * n + 1
            self.H[iA, iA], self.H[iB, iB] = self.delta_arr[n], -self.delta_arr[n]
            self.add_hopping(iA, iB, self.v_arr[n])
            self.add_hopping(iB, 2*(n+1), self.w_arr[n])
            self.add_hopping(iA, 2*(n+1), self.tA)
            self.add_hopping(iB, 2*(n+1)+1, self.tB)
            self.add_hopping(iA, 2*(n+1)+1, self.u1)
            self.add_hopping(iB, 2*(n+2), self.u2)

class DiscretizedContinuumModel(BaseLattice1D):
    def __init__(self, N, pbc=False, profile='tanh', m0=0.5, vF=1.0, chi=5.0, delta=0.0):
        super().__init__(N, pbc)
        x = np.arange(N) - (N / 2)
        if profile == 'tanh': mass_profile = m0 * np.tanh(x / chi)
        elif profile == 'step': mass_profile = m0 * np.sign(x)
        elif profile == 'linear': mass_profile = m0 * (x / chi)
        
        self.v_arr = vF + mass_profile / 2.0
        self.w_arr = vF - mass_profile / 2.0
        self.delta_arr = np.full(N, delta)
        self._build_matrix()

    def _build_matrix(self):
        for n in range(self.N):
            iA, iB = 2 * n, 2 * n + 1
            self.H[iA, iA], self.H[iB, iB] = self.delta_arr[n], -self.delta_arr[n]
            self.add_hopping(iA, iB, self.v_arr[n])
            self.add_hopping(iB, 2*(n+1), self.w_arr[n])

# ==========================================
# 2. FastAPI Web Service
# ==========================================
app = FastAPI()

class LatticeParams(BaseModel):
    N: int; pbc: bool; v: float; w: float; tA: float; tB: float; u1: float; u2: float; delta: float
    dw_hopping: bool; dw_staggering: bool

class ContinuumParams(BaseModel):
    N: int; pbc: bool; profile: str; m0: float; vF: float; chi: float; delta: float

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/api/lattice")
def solve_lattice(params: LatticeParams):
    model = SSHLatticeModel(**params.dict())
    model.solve()
    
    m_eff = params.v - params.w - params.u1 + params.u2
    vF_eff = params.w - params.u1 - 2*params.u2
    
    return {
        "evals": model.evals.tolist(),
        "evecs": (np.abs(model.evecs)**2).tolist(),
        "ipr": model.ipr.tolist(),
        "pol": model.sublattice_pol.tolist(),
        "eff_m": m_eff, "eff_vF": vF_eff,
        "v_arr": model.v_arr.tolist(),
        "w_arr": model.w_arr.tolist(),
        "delta_arr": model.delta_arr.tolist(),
        "N": params.N
    }

@app.post("/api/continuum")
def solve_continuum(params: ContinuumParams):
    model = DiscretizedContinuumModel(**params.dict())
    model.solve()
    return {
        "evals": model.evals.tolist(),
        "evecs": (np.abs(model.evecs)**2).tolist(),
        "ipr": model.ipr.tolist(),
        "pol": model.sublattice_pol.tolist(),
        "v_arr": model.v_arr.tolist(),
        "w_arr": model.w_arr.tolist(),
        "delta_arr": model.delta_arr.tolist(),
        "N": params.N
    }


# ==========================================
# 3. Execution / Debug Block
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        model = SSHLatticeModel(N=60, pbc=True, v=0.8, w=1.2, dw_hopping=True)
        evals, evecs = model.solve()
        print(f"Spectrum computed. Lowest Energy: {np.min(np.abs(evals)):.6f}")
    else:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

        
