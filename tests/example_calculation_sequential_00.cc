// --------------------------------------------------------------------------
// Copyright (C) 2020 by Sebastian Stark
//
// This file is part of the IncrementalFE library
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

//#define PARALLEL

#include <iostream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/conditional_ostream.h>

#include <galerkin_tools/assembly_helper.h>

#include <incremental_fe/fe_model.h>
#include <incremental_fe/scalar_functionals/omega_lib.h>
#include <incremental_fe/scalar_functionals/psi_lib.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

/**
 * postprocessor computing the Cauchy stress from the displacement field
 */
template <int spacedim>
class StressPostprocessor : public DataPostprocessor<spacedim>
{
private:

	/**
	 * Name of stress tensor
	 */
	const string
	name;

	/**
	 * Index of first component of displacement field
	 */
	const unsigned int
	global_component_index_u;

	/**
	 * Index of concentration field
	 */
	const unsigned int
	global_component_index_c;

	/**
	 * Material id for which this produces non-zero graphical output
	 */
	const unsigned int
	material_id;

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	lambda;

	/**
	 * Lame's constant \f$\lambda\f$
	 */
	const double
	mu;

	/**
	 * \f$\Delta \varepsilon\f$
	 */
	const double
	deps;

	/**
	 * \f$c^\mathrm{ref}\f$
	 */
	const double
	c_ref;

public:
	StressPostprocessor(const string&		name,
						const unsigned int	global_component_index_u,
						const unsigned int	global_component_index_c,
						const unsigned int	material_id,
						const double		lambda,
						const double		mu,
						const double		deps,
						const double		c_ref)
	:
	name(name),
	global_component_index_u(global_component_index_u),
	global_component_index_c(global_component_index_c),
	material_id(material_id),
	lambda(lambda),
	mu(mu),
	deps(deps),
	c_ref(c_ref)
	{
	}

	virtual
	void
	evaluate_vector_field(	const DataPostprocessorInputs::Vector<spacedim>&	input_data,
							vector<Vector<double>>&								computed_quantities)
	const override
	{
		const static Tensor<2, 3> I = unit_symmetric_tensor<3,double>();
		static Tensor<2, 3> F, E, T;
		static SymmetricTensor<2,3> sigma, s;
		const auto current_cell = input_data.template get_cell<spacedim>();

		for (unsigned int dataset = 0; dataset < input_data.solution_gradients.size(); ++dataset)
		{

			if(current_cell->material_id() == material_id)
			{
				F = 0.0;
				for (unsigned int d = 0; d < spacedim; ++d)
					for (unsigned int e = 0; e < spacedim; ++e)
						F[d][e] = input_data.solution_gradients[dataset][global_component_index_u + d][e];
				for (unsigned int d = 0; d < 3; ++d)
					F[d][d] += 1.0;

				const double c = input_data.solution_values[dataset][global_component_index_c];
				const double J = determinant(F);
				E = 0.5 * (contract<0, 0>(F, F) - I) - deps/3.0 * (c/c_ref - 1.0) * I;
				T = lambda * trace(E) * I + 2.0 * mu * E;
				sigma = symmetrize(1.0 / J * contract<1,1>(contract<1,0>(F, T), F));
				for (unsigned int d = 0; d < 3; ++d)
					for (unsigned int e = 0; e < 3; ++e)
						computed_quantities[dataset][3 * d + e] = sigma[d][e];
				computed_quantities[dataset][9] = eigenvalues(sigma)[0];
			}
			else
			{
				for (unsigned int d = 0; d < 3; ++d)
					for (unsigned int e = 0; e < 3; ++e)
						computed_quantities[dataset][3 * d + e] = 0.0;
				computed_quantities[dataset][9] = 0.0;
			}
		}
	}

	virtual vector<string>
	get_names()
	const override
	{
		static const char suffixes[] = {'x', 'y', 'z'};
		vector<std::string> names;
		for(unsigned int d = 0; d < 3; ++d)
			for(unsigned int e = 0; e < 3; ++e)
				names.push_back(name + '_' + suffixes[d] + suffixes[e]);
		names.push_back(name + "_max");
		return names;
	}

	virtual vector<DataComponentInterpretation::DataComponentInterpretation>
	get_data_component_interpretation()
	const override
	{
		return vector<DataComponentInterpretation::DataComponentInterpretation>(10, DataComponentInterpretation::component_is_scalar);
	}

	virtual UpdateFlags
	get_needed_update_flags()
	const override
	{
		return update_values | update_gradients;
	}

	virtual
	~StressPostprocessor()
	{
	}
 };


// class defining an initial constant value for a scalar independent field
template<unsigned int spacedim>
class InitialConstant : public Function<spacedim>
{
private:
	const double
	c_0;

public:
	InitialConstant(const double c_0)
	:
	Function<spacedim>(1),
	c_0(c_0)
	{}

	double
	value(	const Point<spacedim>&	/*location*/,
			const unsigned int		/*component*/)
	const
	{
		return c_0;
	}
};

// class used for definition of electrical loading
template<unsigned int spacedim>
class ElectricalLoading : public TotalPotentialContribution<spacedim>
{
private:

	GlobalDataIncrementalFE<spacedim>&
	global_data;

public:

	double
	j_bar = 0.0;

	double
	phi_ap_bar = 0.0;

	double
	R_el = 0.0;

	/**
	 * 0 : prescribed current
	 * 1 : prescribed potential
	 * 2 : prescribed resistance
	 */
	unsigned int
	loading_type = 0;

	ElectricalLoading(	vector<const ScalarFunctional<spacedim, spacedim>*>		H_omega,
						vector<const ScalarFunctional<spacedim-1, spacedim>*>	H_sigma,
						vector<const IndependentField<0, spacedim>*>			C,
						GlobalDataIncrementalFE<spacedim>&						global_data)
	:
	TotalPotentialContribution<spacedim>(H_omega, H_sigma, C),
	global_data(global_data)
	{
	}

	bool get_potential_contribution(const Vector<double>&			H_omega_H_sigma_C,
									const vector<Vector<double>>&	/*C_ref_sets*/,
									double&							Pi,
									dealii::Vector<double>&			Pi_1,
									dealii::FullMatrix<double>&		Pi_2,
									const tuple<bool,bool,bool>&	requested_quantities)
	const
	{
		const double delta_t = global_data.get_t() - global_data.get_t_ref();
		const double phi_ap = H_omega_H_sigma_C[0];
		const double j_ap = H_omega_H_sigma_C[1];

		if(loading_type == 0)
		{
			if(get<0>(requested_quantities))
				Pi = -j_bar * phi_ap * delta_t + 0.5 * j_ap * j_ap;		// the part 0.5 j_ap * j_ap makes sure that j_ap = 0 during this type of loading (j_ap is not needed as unknown)

			if(get<1>(requested_quantities))
			{
				Pi_1[0] = -j_bar * delta_t;
				Pi_1[1] = j_ap;
			}

			if(get<2>(requested_quantities))
			{
				Pi_2(0,0) = 0.0;
				Pi_2(0,1) = Pi_2(1,0) = 0.0;
				Pi_2(1,1) = 1.0;
			}
		}
		else if(loading_type == 1)
		{
			if(get<0>(requested_quantities))
				Pi = -j_ap * (phi_ap - phi_ap_bar) * delta_t;	// enforce the boundary condition by the Lagrangian multiplier j_ap
																// (this has the advantage that the current corresponding to the potential is directly computed as part of the solution)

			if(get<1>(requested_quantities))
			{
				Pi_1[0] = -j_ap * delta_t;
				Pi_1[1] = - (phi_ap - phi_ap_bar) * delta_t;
			}

			if(get<2>(requested_quantities))
			{
				Pi_2(0,0) = 0.0;
				Pi_2(0,1) = Pi_2(1,0) = -delta_t;
				Pi_2(1,1) = 0.0;
			}
		}
		else if(loading_type == 2)
		{
			if(get<0>(requested_quantities))
				Pi = -0.5 / R_el * phi_ap * phi_ap * delta_t  + 0.5 * j_ap * j_ap;		// the part 0.5 j_ap * j_ap makes sure that j_ap = 0 during this type of loading (j_ap is not needed as unknown)

			if(get<1>(requested_quantities))
			{
				Pi_1[0] = -delta_t / R_el * phi_ap;
				Pi_1[1] = j_ap;
			}

			if(get<2>(requested_quantities))
			{
				Pi_2(0,0) = -delta_t / R_el;
				Pi_2(0,1) = Pi_2(1,0) = 0.0;
				Pi_2(1,1) = 1.0;
			}
		}
		else
		{
			Assert(false, ExcMessage("Unknown loading type!"));
		}


		return false;
	}
};


// main function
#ifdef PARALLEL
int main( int argc, char **argv )
#else
int main()
#endif
{

#ifdef PARALLEL
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
#endif

	const unsigned int spacedim = 2;	// this currently doesn't work for anything else than spacedim == 2

/**************
 * parameters *
 **************/

	// quantities used for normalization (in SI units)
	const double c_ast = 10000.0;
	const double D_ast = 1e-16;
	const double R_ast = 8.3144;
	const double T_ast = 293.15;
	const double F_ast = 96485.33;
	const double L_ast = 1e-6;

	// parameters of model
	const double B = 3e-6 / L_ast;
	const double L = 18e-6 / L_ast;
	const double L_ap = 3e-6 / L_ast;

	const double dt_1 = 2700.0 * D_ast / (L_ast * L_ast);

	const double T = 293.15 / T_ast;
	const double c_ap_ref = 47500.0 / c_ast;

	const double lambda_se = 57.7e9 / (R_ast * T_ast * c_ast);
	const double mu_se = 38.5e9 / (R_ast * T_ast * c_ast);
	const double lambda_ap = 50.6e9 / (R_ast * T_ast * c_ast);
	const double mu_ap = 80e9 / (R_ast * T_ast * c_ast);
	const double deps_ap = -0.04;
	const double c_ap_V = 50000.0 / c_ast;
	const double dmu_ap = 70000.0 / (R_ast * T_ast);
	const double D_se = 2.6e-12 / D_ast;
	const double D_ap = 5e-16 / D_ast;
	const double c_se = 10000.0 / c_ast;
	const double i_0_se_ap = 10.0 * L_ast / (F_ast * c_ast * D_ast);
	const double i_0_se_Li = 10.0 * L_ast / (F_ast * c_ast * D_ast);
	const double beta_se_ap = 0.5;
	const double beta_se_Li = 0.5;
	const double eta_bar_Li = 350000.0 / (R_ast * T_ast);
	const double R = 8.3144 / R_ast;
	const double F = 96485.33 / F_ast;

	const double j_ap_bar = -1.0/dt_1 * B * L_ap * (c_ap_ref - (c_ap_V - 0.5 * 0.8 * c_ap_V)) * F;
	const double j_threshold_charging = 0.1;		// percentage of |j_ap_bar| at which constant voltage charging is stopped
	const double j_threshold_discharging = 0.1;		// percentage of |j_ap_bar| at which discharging is stopped

	// numerical parameters
	const double eps_chemical = 1e-4;				// numerical parameter for regularization of chemical potential

	const unsigned int N_se = 5;					// number of elements to be used for the solid electrolyte in coarse mesh
	const unsigned int N_ap = 1;					// number of elements to be used for the active material in coarse mesh
	const unsigned int n_refinements_local = 1;		// number of local refinements at interface between solid electrolyte and active particles
	const unsigned int n_refinements_global = 1;	// number of global refinements

	const double alpha = 1.0;						// time integration parameter alpha
	const unsigned int method = 0;					// time integration method (0: Miehe's method, 1: alpha family, 2: modified alpha family)
	const unsigned N_1 = 20;						// nominal number of time steps for first loading step (automatic time stepping adjusts of necessary)

	const unsigned int degree = 2;					// degree of approximation of finite elements

	// mappings
	MappingQGeneric<spacedim, spacedim> mapping_domain(1);		// FE mapping on domain
	MappingQGeneric<spacedim-1, spacedim> mapping_interface(1);	// FE mapping on interfaces


/**********************
 * independent fields *
 **********************/

	InitialConstant<spacedim> c_ap_initial(c_ap_ref);

	IndependentField<spacedim, spacedim> u("u", FE_Q<spacedim>(degree), spacedim, {0,1});				// displacement field (region 0 is solid electrolyte, region 1 is active material)
	IndependentField<spacedim, spacedim> c_ap("c_ap", FE_Q<spacedim>(degree), 1, {1}, &c_ap_initial);	// Lithium concentration in active material
	IndependentField<spacedim, spacedim> eta_ap("eta_ap", FE_Q<spacedim>(degree), 1, {1});				// chemical potential of Li in active material
	IndependentField<spacedim, spacedim> eta_se("eta_se", FE_Q<spacedim>(degree), 1, {0});				// electrochemical potential of Li ions in solid electrolyte
	IndependentField<0, spacedim> phi_ap("phi_ap");														// electrical scalar potential of active particles
	IndependentField<0, spacedim> j_ap("j_ap");															// electrical current into active particles
	IndependentField<0, spacedim> constant_displacement("constant_displacement");						// constant displacement for periodic b.c.

/********************
 * dependent fields *
 ********************/

	// deformation gradient in solid electrolyte and active material
	DependentField<spacedim, spacedim> F_xx("F_xx");
	DependentField<spacedim, spacedim> F_yy("F_yy");
	DependentField<spacedim, spacedim> F_zz("F_zz");
	DependentField<spacedim, spacedim> F_xy("F_xy");
	DependentField<spacedim, spacedim> F_yz("F_yz");
	DependentField<spacedim, spacedim> F_zx("F_zx");
	DependentField<spacedim, spacedim> F_yx("F_yx");
	DependentField<spacedim, spacedim> F_zy("F_zy");
	DependentField<spacedim, spacedim> F_xz("F_xz");
	F_xx.add_term(1.0, u, 0, 0);
	F_xx.add_term(1.0);
	F_yy.add_term(1.0, u, 1, 1);
	F_yy.add_term(1.0);
	F_xy.add_term(1.0, u, 0, 1);
	F_yx.add_term(1.0, u, 1, 0);
	F_zz.add_term(1.0);
	if(spacedim == 3)
	{
		F_zz.add_term(1.0, u, 2, 2);
		F_yz.add_term(1.0, u, 1, 2);
		F_zy.add_term(1.0, u, 2, 1);
		F_zx.add_term(1.0, u, 2, 0);
		F_xz.add_term(1.0, u, 0, 2);
	}

	// Lithium ion concentration in solid electrolyte
	DependentField<spacedim, spacedim> c_se_("c_se_");
	c_se_.add_term(c_se);

	// Lithium concentration in active material
	DependentField<spacedim, spacedim> c_ap_("c_ap_");
	c_ap_.add_term(1.0, c_ap);

	// vacancy concentration in active material
	DependentField<spacedim, spacedim> c_ap_V_("c_ap_V");
	c_ap_V_.add_term(-1.0, c_ap);
	c_ap_V_.add_term(c_ap_V);

	// chemical potential of Li in active material
	DependentField<spacedim, spacedim> eta_ap_("eta_ap_");
	DependentField<spacedim, spacedim> eta_ap_x("eta_ap_x");
	DependentField<spacedim, spacedim> eta_ap_y("eta_ap_y");
	DependentField<spacedim, spacedim> eta_ap_z("eta_ap_z");
	eta_ap_.add_term(1.0, eta_ap);
	eta_ap_x.add_term(1.0, eta_ap, 0, 0);
	eta_ap_y.add_term(1.0, eta_ap, 0, 1);
	if(spacedim == 3)
		eta_ap_z.add_term(1.0, eta_ap, 0, 2);

	// electrochemical potential of Li ions in solid electrolyte
	DependentField<spacedim, spacedim> eta_se_("eta_se_");
	DependentField<spacedim, spacedim> eta_se_x("eta_se_x");
	DependentField<spacedim, spacedim> eta_se_y("eta_se_y");
	DependentField<spacedim, spacedim> eta_se_z("eta_se_z");
	eta_se_.add_term(1.0, eta_se);
	eta_se_x.add_term(1.0, eta_se, 0, 0);
	eta_se_y.add_term(1.0, eta_se, 0, 1);
	if(spacedim == 3)
		eta_se_z.add_term(1.0, eta_se, 0, 2);

	// potential difference on solid electrolyte - Lithium interface
	DependentField<spacedim-1, spacedim> deta_se_Li("deta_se_Li");
	deta_se_Li.add_term(-1.0, eta_se, 0, InterfaceSide::minus);
	deta_se_Li.add_term(eta_bar_Li);

	// potential difference on solid electrolyte - active material interface
	DependentField<spacedim-1, spacedim> deta_se_ap("deta_se_ap");
	deta_se_ap.add_term(-1.0, eta_se, 0, InterfaceSide::minus);
	deta_se_ap.add_term(1.0, eta_ap, 0, InterfaceSide::plus);
	deta_se_ap.add_term(F, phi_ap);

/********
 * grid *
 ********/

#ifdef PARALLEL
	dealii::parallel::distributed::Triangulation<spacedim> tria_domain_se (MPI_COMM_WORLD), tria_domain_ap(MPI_COMM_WORLD), tria_domain(MPI_COMM_WORLD);
#else
	dealii::Triangulation<spacedim> tria_domain_se, tria_domain_ap, tria_domain;
#endif

	// generate coarse mesh of domain
	GridGenerator::subdivided_hyper_rectangle(tria_domain_se, {N_se, 1}, Point<spacedim>(0.0, 0.0), Point<spacedim>(L - L_ap, B));
	GridGenerator::subdivided_hyper_rectangle(tria_domain_ap, {N_ap, 1}, Point<spacedim>(L - L_ap, 0.0), Point<spacedim>(L, B));
	GridGenerator::merge_triangulations(tria_domain_se, tria_domain_ap, tria_domain);

	// define domains
	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		// material id of solid electrolyte = 0
		if(cell->center()[0] < L - L_ap)
			cell->set_material_id(0);
		// material id of active material = 1
		else
			cell->set_material_id(1);
	}

	// mesh refinement
	for(unsigned int refinement_step = 0; refinement_step < n_refinements_local; ++refinement_step)
	{
		for(const auto& cell : tria_domain.active_cell_iterators())
		{
			for(unsigned int face = 0; face<GeometryInfo<spacedim>::faces_per_cell; ++face)
			{
				if( ( (cell->material_id() == 1) && (cell->face(face)->center()[0] < (L - L_ap) + 1e-12) )
						||
					( (cell->material_id() == 0) && (cell->face(face)->center()[0] > (L - L_ap) - 1e-12) )
				  )
					cell->set_refine_flag();
			}
		}
		tria_domain.execute_coarsening_and_refinement();
	}
	tria_domain.refine_global(n_refinements_global);

	// triangulation system
	// includes interface definition (	left interface to environment: 0,
	//									right interface to environment: 1,
	//									bottom interface to environment: 2,
	//									top interface to environment: 3,
	//									interface between solid electrolyte and active material: 4 )
#ifdef PARALLEL
	dealii::GalerkinTools::parallel::TriangulationSystem<spacedim> tria_system(tria_domain);
#else
	dealii::GalerkinTools::TriangulationSystem<spacedim> tria_system(tria_domain);
#endif

	for(const auto& cell : tria_domain.cell_iterators_on_level(0))
	{
		for(unsigned int face = 0; face < GeometryInfo<spacedim>::faces_per_cell; ++face)
		{
			if(cell->face(face)->at_boundary())
			{
				if(cell->face(face)->center()[0] < 1e-12)
					tria_system.add_interface_cell(cell, face, 0);
				else if(cell->face(face)->center()[0] > L - 1e-12)
					tria_system.add_interface_cell(cell, face, 1);
				else if(cell->face(face)->center()[1] < 1e-12)
					tria_system.add_interface_cell(cell, face, 2);
				else if(cell->face(face)->center()[1] > B - 1e-12)
					tria_system.add_interface_cell(cell, face, 3);
			}
			else
				if( (cell->material_id() == 0) && (cell->face(face)->center()[0] > (L - L_ap) - 1e-12) )
					tria_system.add_interface_cell(cell, face, 4);
		}
	}
	tria_system.close();

/*************
 * potential *
 *************/

	// global data object for information transfer between different places
	GlobalDataIncrementalFE<spacedim> global_data;

	// psi_se
	KirchhoffMaterial00<spacedim> psi_se(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_se_},
											{0},
											QGauss<spacedim>(degree + 1),
											global_data,
											lambda_se,
											mu_se,
											0.0,
											1.0,
											alpha);

	// psi_ap_e
	KirchhoffMaterial00<spacedim> psi_ap_e(	{F_xx, F_xy, F_xz, F_yx, F_yy, F_yz, F_zx, F_zy, F_zz, c_ap_},
												{1},
												QGauss<spacedim>(degree + 1),
												global_data,
												lambda_ap,
												mu_ap,
												deps_ap,
												c_ap_ref,
												alpha);

	// psi_ap_c (part 1)
	PsiChemical00<spacedim> psi_ap_c_1(	{c_ap_},
										{1},
										QGaussLobatto<spacedim>(degree + 1),
										global_data,
										R*T, c_ap_ref, 0.0,
										alpha,
										eps_chemical);

	// psi_ap_c (part 2)
	PsiChemical00<spacedim> psi_ap_c_2(	{c_ap_V_},
										{1},
										QGaussLobatto<spacedim>(degree + 1),
										global_data,
										R*T, c_ap_V - c_ap_ref, 0.0,
										alpha,
										eps_chemical);

	// psi_ap_c (part 3)
	PsiChemical01<spacedim> psi_ap_c_3(	{c_ap_},
										{1},
										QGaussLobatto<spacedim>(degree + 1),
										global_data,
										dmu_ap/c_ap_V, c_ap_ref,
										alpha);


	// delta_se
	OmegaDualFluxDissipation00<spacedim> delta_se(	{eta_se_x, eta_se_y, eta_se_z, c_se_},
													{0},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_se/(R*T),
													method,
													alpha);

	// delta_ap
	OmegaDualFluxDissipation00<spacedim> delta_ap(	{eta_ap_x, eta_ap_y, eta_ap_z, c_ap_},
													{1},
													QGauss<spacedim>(degree + 1),
													global_data,
													D_ap/(R*T),
													method,
													alpha);

	// mixed term for active material
	OmegaMixedTerm00<spacedim> omega_mixed_ap(	{c_ap_, eta_ap_},
												{1},
												QGauss<spacedim>(degree + 1),
												global_data,
												method,
												alpha);

	// delta_se_Li
	OmegaDualButlerVolmer00<spacedim> delta_se_Li(	{deta_se_Li},
													{0},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_Li * R * T / F, beta_se_Li, R * T, 20.0,
													method,
													alpha);

	// delta_se_ap
	OmegaDualButlerVolmer00<spacedim> delta_se_ap(	{deta_se_ap},
													{4},
													QGauss<spacedim-1>(degree + 1),
													global_data,
													i_0_se_ap * R * T / F, beta_se_ap, R * T, 20.0,
													method,
													alpha);

	// total potential contributions
	TotalPotentialContribution<spacedim> psi_se_tpc(psi_se);
	TotalPotentialContribution<spacedim> psi_ap_e_tpc(psi_ap_e);
	TotalPotentialContribution<spacedim> psi_ap_c_1_tpc(psi_ap_c_1);
	TotalPotentialContribution<spacedim> psi_ap_c_2_tpc(psi_ap_c_2);
	TotalPotentialContribution<spacedim> psi_ap_c_3_tpc(psi_ap_c_3);
	TotalPotentialContribution<spacedim> delta_se_tpc(delta_se);
	TotalPotentialContribution<spacedim> delta_ap_tpc(delta_ap);
	TotalPotentialContribution<spacedim> omega_mixed_ap_tpc(omega_mixed_ap);
	TotalPotentialContribution<spacedim> delta_se_Li_tpc(delta_se_Li);
	TotalPotentialContribution<spacedim> delta_se_ap_tpc(delta_se_ap);
	vector<const ScalarFunctional<spacedim,spacedim>*> H_omega_electrical_loading;
	vector<const ScalarFunctional<spacedim-1,spacedim>*> H_sigma_electrical_loading;
	vector<const IndependentField<0,spacedim>*> C_electrical_loading;
	C_electrical_loading.push_back(&phi_ap);
	C_electrical_loading.push_back(&j_ap);
	ElectricalLoading<spacedim> electrical_loading_tpc(H_omega_electrical_loading, H_sigma_electrical_loading, C_electrical_loading, global_data);

	// total potential
	TotalPotential<spacedim> total_potential;
 	total_potential.add_total_potential_contribution(psi_se_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_e_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_1_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_2_tpc);
 	total_potential.add_total_potential_contribution(psi_ap_c_3_tpc);
 	total_potential.add_total_potential_contribution(delta_se_tpc);
	total_potential.add_total_potential_contribution(delta_ap_tpc);
	total_potential.add_total_potential_contribution(omega_mixed_ap_tpc);
	total_potential.add_total_potential_contribution(delta_se_Li_tpc);
	total_potential.add_total_potential_contribution(delta_se_ap_tpc);
	total_potential.add_total_potential_contribution(electrical_loading_tpc);


	// add constraints
	// u_x
	DirichletConstraint<spacedim> dc_u_x(u, 0, InterfaceSide::minus, {0});
	// u_y
	DirichletConstraint<spacedim> dc_u_y_bottom(u, 1, InterfaceSide::minus, {2});
	DirichletConstraint<spacedim> dc_u_y_top(u, 1, InterfaceSide::minus, {3}, nullptr, &constant_displacement);

	Constraints<spacedim> constraints;
	constraints.add_dirichlet_constraint(dc_u_x);
	constraints.add_dirichlet_constraint(dc_u_y_bottom);
	constraints.add_dirichlet_constraint(dc_u_y_top);

/***************************************************
 * set up finite element model and do computations *
 ***************************************************/

#ifdef PARALLEL
	SolverWrapperPETSc solver_wrapper;
	FEModel<spacedim, LinearAlgebra::distributed::Vector<double>, PETScWrappers::MPI::BlockVector, GalerkinTools::parallel::TwoBlockMatrix<PETScWrappers::MPI::SparseMatrix>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper);
#else
	BlockSolverWrapperUMFPACK solver_wrapper;
	FEModel<spacedim, Vector<double>, BlockVector<double>, GalerkinTools::TwoBlockMatrix<SparseMatrix<double>>> fe_model(total_potential, tria_system, mapping_domain, mapping_interface, global_data, constraints, solver_wrapper);
#endif

	const unsigned int this_proc = fe_model.get_assembly_helper().get_triangulation_system().get_this_proc_n_procs().first;
	ConditionalOStream pout(cout, this_proc == 0);
	global_data.set_output_level(0);

	// postprocessors
	StressPostprocessor<spacedim> stress_postproc_se(	"sigma_se",
														fe_model.get_assembly_helper().get_u_omega_global_component_index(u),
														fe_model.get_assembly_helper().get_u_omega_global_component_index(c_ap),	// doesn't matter
														0,
														lambda_se,
														mu_se,
														0.0,
														c_se);
	fe_model.attach_data_postprocessor_domain(stress_postproc_se);

	StressPostprocessor<spacedim> stress_postproc_ap(	"sigma_ap",
														fe_model.get_assembly_helper().get_u_omega_global_component_index(u),
														fe_model.get_assembly_helper().get_u_omega_global_component_index(c_ap),	// doesn't matter
														1,
														lambda_ap,
														mu_ap,
														deps_ap,
														c_ap_ref);
	fe_model.attach_data_postprocessor_domain(stress_postproc_ap);

	const unsigned int dof_index_phi_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&phi_ap);
	const unsigned int dof_index_j_ap = fe_model.get_assembly_helper().get_global_dof_index_C(&j_ap);

	// vector for (t, phi_ap, j_ap)
	vector<tuple<double, double, double>> phi_j;

	// iteration settings
	global_data.set_max_iter(20);
	global_data.set_max_cutbacks(1);
	global_data.set_threshold_potential_increment(1e-7);

// first step
	electrical_loading_tpc.loading_type = 0;
	electrical_loading_tpc.j_bar = j_ap_bar;
	double start_time = global_data.get_t();
	double end_time = start_time + dt_1;
	unsigned int n_increments_initial = N_1;
	double inc = (end_time - start_time)/(double)n_increments_initial;
	double inc_initial = inc;
	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		if(new_time > end_time)
			new_time = end_time;
		pout << "Old time=" << global_data.get_t() << endl;
		pout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), j_ap_bar));
			if(new_time == end_time)
				break;
			if(iter < 4)
			{
				inc = inc * 2.0;
				if(inc > inc_initial)
					inc = inc_initial;
			}
		}
		else
		{
			pout << "ERROR, CUTBACK necessary!" << endl;
			inc = inc / 2.0;
			continue;
		}
		pout << endl;
	}
	pout << endl;

// second step
	electrical_loading_tpc.loading_type = 1;
	electrical_loading_tpc.phi_ap_bar = fe_model.get_solution_vector()(dof_index_phi_ap);
	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		pout << "Old time=" << global_data.get_t() << endl;
		pout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), fe_model.get_solution_vector()(dof_index_j_ap)));
			if( fabs(fe_model.get_solution_vector()(dof_index_j_ap)) <= fabs(j_threshold_charging * j_ap_bar) )
				break;
			if(iter < 4)
			{
				inc = inc * 2.0;
				if(inc > inc_initial)
					inc = inc_initial;
			}
		}
		else
		{
			pout << "ERROR, CUTBACK necessary!" << endl;
			inc = inc / 2.0;
			continue;
		}
		pout << endl;
	}
	pout << endl;

// third step
	electrical_loading_tpc.loading_type = 2;
	electrical_loading_tpc.R_el = fabs(fe_model.get_solution_vector()(dof_index_phi_ap) / (2.0 * j_ap_bar));
	for(;;)
	{
		double new_time = global_data.get_t() + inc;
		pout << "Old time=" << global_data.get_t() << endl;
		pout << "New time=" << new_time << endl;

		const int iter = fe_model.do_time_step(new_time);
		if(iter >= 0)
		{
			phi_j.push_back(make_tuple(new_time, fe_model.get_solution_vector()(dof_index_phi_ap), -fe_model.get_solution_vector()(dof_index_phi_ap)/electrical_loading_tpc.R_el));
			if( fabs(fe_model.get_solution_vector()(dof_index_phi_ap)/electrical_loading_tpc.R_el) <= fabs(j_threshold_discharging * j_ap_bar) )
				break;
			if(iter < 4)
			{
				inc = inc * 2.0;
				if(inc > inc_initial)
					inc = inc_initial;
			}
		}
		else
		{
			pout << "ERROR, CUTBACK necessary!" << endl;
			inc = inc / 2.0;
			continue;
		}
		pout << endl;
	}
	pout << endl;

	for(const auto& phi_el : phi_j)
		pout << get<0>(phi_el) << " " << get<1>(phi_el) << " " << get<2>(phi_el) << endl;
}
