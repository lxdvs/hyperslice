Terminology for Valid and Invalid Regions in Rectilinear Parameter Spaces

Motivation

Many scientific and engineering parameter sweeps are performed over a full-factorial, rectilinear design space. While the underlying design is a complete Cartesian product of parameter values, not every sampled point necessarily represents a valid physical state.

Examples include:

* Solver non-convergence
* Physically impossible operating conditions
* Safety limit violations
* User-excluded operating regions
* Numerical failures

Treating every invalid point identically is often misleading. There is an important distinction between invalid regions that represent the exterior of the physically feasible design space and isolated “holes” embedded within an otherwise valid region.

This document defines terminology for describing these situations.

⸻

Design Space

The design space is the Cartesian product of every coordinate value along every parameter axis.

For example:

temperature = [600, 700, 850, 1000]
pressure    = [1, 2, 5]
flow         = [2, 5]

defines a rectilinear design space containing

4 × 3 × 2 = 24

possible design points.

The design space is independent of whether every point is valid.

⸻

Rectilinear Grid

A rectilinear grid is one in which each parameter axis is represented by an independent one-dimensional coordinate array.

Rectilinear grids do not require uniform spacing.

For example:

temperature = [600, 700, 850, 1200]
pressure    = [1, 2, 4, 8, 16]

is still rectilinear.

⸻

Complete and Incomplete Datasets

A complete rectilinear dataset contains an evaluated value for every point in the Cartesian product.

An incomplete rectilinear dataset omits one or more Cartesian combinations.

The underlying design remains rectilinear even if the sampled dataset is incomplete.

⸻

Valid Point

A valid point is a design point that:

* was evaluated successfully,
* represents a physically meaningful state,
* satisfies all imposed constraints.

⸻

Invalid Point

An invalid point is a design point that does not produce a usable result.

The reason may include:

* solver failure
* non-convergence
* physical infeasibility
* excluded operating condition
* user filtering

The reason should ideally be stored separately from the validity itself.

⸻

Invalid Region

An invalid region is a connected component of invalid points.

Connectivity is defined using the chosen neighborhood rule (for example, face-connected neighbors in N dimensions).

Working with connected regions is generally more useful than reasoning about isolated invalid points.

⸻

Boundary-Connected Invalid Region

A boundary-connected invalid region is an invalid connected component that intersects the boundary of the design space.

Equivalently:

An invalid region is boundary-connected if at least one invalid point within the region lies on the boundary of the sampled design space.

Example:

████████
██████..
█████...
████....
██......

Every invalid point belongs to the same boundary-connected region.

These regions frequently represent:

* physically infeasible operating conditions,
* operating points outside the feasible envelope,
* regions beyond safety limits.

Interpolation should generally not cross these regions.

⸻

Interior Invalid Region

An interior invalid region (or invalid island) is an invalid connected component that does not intersect the boundary of the design space.

Example:

████████
██..████
██..████
████████

Interior regions frequently represent:

* isolated solver failures,
* missing simulations,
* numerical instability,
* corrupted data.

These regions may be appropriate candidates for interpolation or repair.

⸻

Boundary Distance

Not every exterior region literally touches the sampled boundary.

Consider:

████████
████████
████████
██..████

Although the invalid component does not intersect the boundary, it is intuitively much more similar to an exterior invalid region than to an isolated interior hole.

To capture this distinction, define the boundary distance of an invalid region.

Definition

The boundary distance is the minimum graph distance from any point in an invalid connected component to any boundary point of the design space.

Properties:

* Boundary-connected regions have distance 0.
* Regions one grid step away from the boundary have distance 1.
* Larger values indicate increasingly interior regions.

This provides a continuous measure instead of a strict binary classification.

⸻

Near-Boundary Invalid Region

A near-boundary invalid region is an invalid connected component whose boundary distance is less than or equal to a chosen threshold.

For example,

boundary distance ≤ 1

may be considered exterior for many engineering applications.

The threshold is application-dependent.

⸻

Exterior Invalid Region

An exterior invalid region is an invalid connected component classified as representing the exterior of the feasible operating domain.

A practical implementation may classify regions as exterior when:

* they are boundary-connected, or
* they lie within a configurable boundary-distance threshold.

Exterior regions usually indicate:

* impossible operating conditions,
* unsafe parameter combinations,
* outside the admissible design space.

Interpolation should generally terminate at the boundary of these regions rather than bridging across them.

⸻

Interior Invalid Island

An interior invalid island is an isolated invalid connected component that is sufficiently separated from the design-space boundary.

These often represent:

* failed simulations,
* isolated missing data,
* numerical artifacts.

Because they are embedded within otherwise valid neighborhoods, they may be candidates for interpolation, resampling, or repair.

⸻

Suggested Classification

Each invalid connected component can be characterized by the following properties.

Property	Meaning
Component ID	Unique connected-component identifier
Size	Number of invalid points
Boundary Connected	Whether the component touches the design-space boundary
Boundary Distance	Minimum graph distance to the boundary
Bounding Box	Axis-aligned bounds
Centroid	Center of the component
Classification	Exterior, Near-Boundary, Interior Island, Unknown

⸻

Recommended Status Categories

Validity and failure reason should remain separate concepts.

Example status values:

Status	Meaning
Valid	Successfully evaluated
Not Evaluated	Never sampled
Non-Converged	Solver failed to converge
Physically Invalid	Violates physical constraints
Solver Failure	Numerical failure
User Excluded	Explicitly omitted

Region classification is then computed independently from these statuses.

⸻

Interpolation Policy

Different invalid-region classes should be treated differently.

Exterior Regions

Interpolation should generally not cross exterior invalid regions.

These regions typically represent genuine discontinuities in the feasible operating space.

⸻

Interior Islands

Small interior islands may be appropriate candidates for:

* local interpolation,
* surrogate modeling,
* targeted resampling,
* solver retry.

The interpolation policy should remain configurable.

⸻

Summary

HyperSlice distinguishes between point validity and region topology.

Individual points are labeled according to why they are invalid.

Connected invalid regions are then classified according to their geometric relationship with the design-space boundary.

This distinction enables more intelligent visualization, interpolation, and analysis than treating every invalid point as equivalent.

The recommended terminology is:

* Valid Point
* Invalid Point
* Invalid Region
* Boundary-Connected Invalid Region
* Boundary Distance
* Near-Boundary Invalid Region
* Exterior Invalid Region
* Interior Invalid Region
* Interior Invalid Island

These concepts are dimension-independent and apply equally to two-dimensional contour plots, three-dimensional parameter spaces, and arbitrary N-dimensional rectilinear datasets.