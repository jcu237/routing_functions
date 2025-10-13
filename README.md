# ConnectedComponents.jl

**ConnectedComponents.jl** is a Julia package for studying the **connectivity of real algebraic varieties** using *routing functions*.  

The package implements algorithms from  
> **"Smooth Connectivity in Real Algebraic Varieties"**  
> *Joseph Cummings, Jonathan Hauenstein, Hoon Hong, and Clifford Smyth*
> *Numerical Algorithms, 100(1), 63-84, 2025*

This paper introduces the use of **routing functions** — rational functions whose gradient flows reveal the connected components of a real algebraic variety.  
`ConnectedComponents.jl` automates this process, providing an efficient framework for computing critical points and tracking gradient paths.

Several examples are included in 'testing.jl'. 

Comments and suggestions are welcome!

Things that need updating:
> 1) The **path_tracking.jl** needs serious consideration. Currently, the step-size is fixed (and can be changed by the user); however, this should probably depend on the eigenvalues of the Hessian or something. The optimization folks must know the answer.
> 2) Currently, to find the critical points, the package is making a call to HC.jl's **solve**. In **HypersurfaceRegions.jl**, they use a monodromy solve. Can this be implemented more generally? This does seem to be the main bottleneck.
