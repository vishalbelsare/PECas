import casadi as ca
import casadi.tools as cat

class BPEval:

    def __init__(self, bp, N):

        self.Y = []
        self.G = []

        self.V = cat.struct_symMX([
                (
                    cat.entry("T", repeat = [N], shape = bp.v["t"].shape),
                    cat.entry("U", repeat = [N], shape = bp.v["u"].shape),
                    cat.entry("P", shape = bp.v["p"].shape),
                    cat.entry("W", repeat = [N], shape = bp.fcn["g"].shape)
                )
            ])

        yfcn = ca.SXFunction([bp.v["t"], bp.v["u"], bp.v["p"]], [bp.fcn["y"]])
        yfcn.init()

        gfcn = ca.SXFunction([bp.v["t"], bp.v["u"], bp.v["p"]], [bp.fcn["g"]])
        gfcn.init()

        for k in range(N):

            self.Y.append(yfcn.call([self.V["T"][k], self.V["U"][k], \
                self.V["P"]])[0])
            self.G.append(gfcn.call([self.V["T"][k], self.V["U"][k], \
                self.V["P"]])[0])

        self.Y = ca.vertcat(self.Y)
        self.G = ca.vertcat(self.G)


class ODESolSingleShooting:

    def __init__(self, op, N, x0):

        self.Y = []
        self.G = []

        self.V = cat.struct_MX([
                (
                    cat.entry("T", repeat = [N], shape = op.v["t"].shape),
                    cat.entry("U", repeat = [N], shape = op.v["u"].shape),
                    cat.entry("X", repeat = [N], shape = op.v["x"].shape),
                    cat.entry("P", shape = op.v["p"].shape),
                    cat.entry("W", repeat = [N], shape = op.fcn["g"].shape)
                )
            ])

        ffcn = ca.SXFunction([op.v["t"], op.v["u"], op.v["x"], op.v["p"]],
            [op.fcn["f"]])
        ffcn.init()

        # self.V["X", 0, :] = x0

        for k in range(N-1):

            [K1] = ffcn.call([self.V["T", k], self.V["U", k, :], self.V["X", k, :], self.V["P", :]])
            [K2] = ffcn.call([self.V["T", k], self.V["U", k, :], self.V["X", k, :] + 0.5 * K1 * (self.V["T", k+1] - self.V["T", k]), self.V["P", :]])
            [K3] = ffcn.call([self.V["T", k], self.V["U", k, :], self.V["X", k, :] + 0.5 * K2 * (self.V["T", k+1] - self.V["T", k]), self.V["P", :]])
            [K4] = ffcn.call([self.V["T", k], self.V["U", k, :], self.V["X", k, :] + K3 * (self.V["T", k+1] - self.V["T", k]), self.V["P", :]])

            self.V["X", k+1, :] = (1.0 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4) * (self.V["T", k+1] - self.V["T", k])

            # self.V["X", k + 1, :] = ca.integratorOut(integrator(ca.integratorIn([self.V["T", k], self.V["U", k, :], self.V["X", k, :], self.V["P", :]])))
