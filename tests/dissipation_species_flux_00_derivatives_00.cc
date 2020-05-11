#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/dissipation_species_flux_00.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{

	//i
	IndependentField<spacedim, spacedim> i("i", FE_Nothing<spacedim>(), spacedim, {});
	//c
	IndependentField<spacedim, spacedim> c("c", FE_Nothing<spacedim>(), 1, {});


	vector<DependentField<spacedim, spacedim>> dependent_fields;

	DependentField<spacedim,   spacedim> i_x("i_x");
	i_x.add_term(1.0, i, 0);

	DependentField<spacedim,   spacedim> i_y("i_y");
	i_y.add_term(1.0, i, 1);

	DependentField<spacedim,   spacedim> i_z("i_z");
	if(spacedim==3)
		i_z.add_term(1.0, i, 2);

	DependentField<spacedim,   spacedim> c_("c");
		c_.add_term(1.0, c);

	dependent_fields.push_back(i_x);
	dependent_fields.push_back(i_y);
	dependent_fields.push_back(i_z);
	dependent_fields.push_back(c_);

	const double D = 0.9;
	const double c0 = 0.2;
	const double alpha = 0.5;
	const unsigned int method = 0;
	const unsigned int sym_mode = 0;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(0.5);
	global_data.set_t(1.5);

	DissipationSpeciesFlux00<spacedim> dissipation_species_flux(dependent_fields,
																{},
																QGauss<spacedim>(1),
																global_data,
																D,
																c0,
																alpha,
																method,
																sym_mode);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	const double N = e_omega.size();
	for(unsigned int m = 0; m < e_omega.size(); ++m)
	{
		e_omega[m] = cos( 2.0*numbers::PI*((double)(m+1))/N ) + 1.1;
		e_omega_ref_sets[0][m] = sin( 2.0*numbers::PI*((double)(m+1))/N ) + 1.1;
	}
	dissipation_species_flux.compare_derivatives_with_numerical_derivatives(e_omega,
																			e_omega_ref_sets,
																			hidden_vars,
																			x);
}


int main()
{
	cout << "### 2D-Case ###\n\n";
	check<2>();

	cout << "\n### 3D-Case ###\n\n";
	check<3>();
}
