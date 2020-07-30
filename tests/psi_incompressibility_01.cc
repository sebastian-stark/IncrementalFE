#include <iostream>
#include <math.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/psi_lib.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
void
check()
{

	vector<DependentField<spacedim, spacedim>> dependent_fields(7, DependentField<spacedim, spacedim>("q"));

	const double alpha = 0.5;

	GlobalDataIncrementalFE<spacedim> global_data;
	global_data.set_t(1.0);

	PsiIncompressibility01<spacedim> psi_incompressibility_01(	dependent_fields,
																{},
																QGauss<spacedim>(1),
																global_data,
																alpha);

	Vector<double> e_omega(dependent_fields.size());
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(dependent_fields.size());
	Vector<double> hidden_vars;
	Point<spacedim> x;

	double a = 0.1;
	for(auto& val : e_omega)
	{
		val = cos(a);
		a = a + 0.1;
	}
	e_omega[0] += 1.0;
	e_omega[3] += 1.0;
	e_omega[5] += 1.0;

	for(auto& val : e_omega_ref_sets[0])
	{
		val = sin(a);
		a = a + 0.1;
	}
	e_omega_ref_sets[0][0] += 1.0;
	e_omega_ref_sets[0][3] += 1.0;
	e_omega_ref_sets[0][5] += 1.0;

	psi_incompressibility_01.compare_derivatives_with_numerical_derivatives(e_omega,
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
