#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/chemical_potential_01.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{

	vector<DependentField<spacedim, spacedim>> dependent_fields(10, DependentField<spacedim, spacedim>("q"));

	const double RT = 1.5;
	const double c0 = 0.2;
	const double mu0 = 1.3;
	const double alpha = 0.5;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	ChemicalPotential01<spacedim> chemical_potential(	dependent_fields,
														{},
														QGauss<spacedim>(1),
														global_data,
														RT,
														c0,
														mu0,
														alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;
	const double N = e_omega.size();
	for(unsigned int m = 0; m < e_omega.size(); ++m)
	{
		e_omega[m] = - cos( 2.0*numbers::PI*((double)(m+1))/N ) - 1.1;
		e_omega_ref_sets[0][m] = - sin( 2.0*numbers::PI*((double)(m+1))/N ) - 1.1;
	}
	e_omega[0] *= -1.0;
	e_omega_ref_sets[0][0] *= -1.0;

	chemical_potential.compare_derivatives_with_numerical_derivatives(	e_omega,
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
