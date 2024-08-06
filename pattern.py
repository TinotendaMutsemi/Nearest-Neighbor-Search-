class Pattern:

    def __init__(self):
        self.b = None  # aggregated controls (T,M) Time x Objects
        self.S = None  # aggregated performance (T,1) Time x 1
        self.h = None  # agent controls (N,M,T) Agents x Objects x Time
        self.SH = None  # agents performance (T,N) Time x Agents

        # Private properties
        self._x = None
        self._k = None
        self._ell = None
        self._s = None
        self._ci = None
        self._mnp = None
        self._ntype = 'active'
        self._qH = None
        self._xnk = {}
        self._horz = 1
        self._lntype = 'absolute'
        self._ltype = 'univlearning'
        self._lparam = None
        self._ptype = 'trivial'
        self._pl = None
        self._px = None
        self._pidx = None
        self._sitype = None
        self._siidx = None
