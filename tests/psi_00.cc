#include <iostream>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/quadrature_lib.h>

#include <incremental_fe/scalar_functionals/psi.h>

using namespace std;
using namespace dealii;
using namespace dealii::GalerkinTools;
using namespace incrementalFE;

template<unsigned int spacedim>
class PsiDomain : public incrementalFE::Psi<spacedim, spacedim>
{

private:

	FullMatrix<double>
	A;

	Vector<double>
	b;

public:

	PsiDomain(	const std::vector<dealii::GalerkinTools::DependentField<spacedim,spacedim>>	e_omega,
				const std::set<dealii::types::material_id>									domain_of_integration,
				const dealii::Quadrature<spacedim>											quadrature,
				GlobalDataIncrementalFE<spacedim>&											global_data,
				const FullMatrix<double>													A,
				const Vector<double>														b,
				const double																alpha = 0.0,
				const std::string															name = "Omega")
	:
	Psi<spacedim, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, name),
	A(A),
	b(b)
	{
	}

	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const unsigned int N = A.m();
		Vector<double> A_q(N);
		A.vmult(A_q, values);

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * (values * A_q ) + b * values;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < N; ++m)
				d_omega[m] = A_q[m] + b[m];
		}

		if(get<2>(requested_quantities))
		{
			d2_omega = A;
		}

		return false;
	}

};

template<unsigned int spacedim>
class PsiInterface : public incrementalFE::Psi<spacedim-1, spacedim>
{

private:

	FullMatrix<double>
	A;

	Vector<double>
	b;

public:

	PsiInterface(	const std::vector<dealii::GalerkinTools::DependentField<spacedim-1,spacedim>>	e_omega,
					const std::set<dealii::types::material_id>										domain_of_integration,
					const dealii::Quadrature<spacedim-1>											quadrature,
					GlobalDataIncrementalFE<spacedim>&												global_data,
					const FullMatrix<double>														A,
					const Vector<double>															b,
					const double																	alpha = 0.0,
					const std::string																name = "Omega")
	:
	Psi<spacedim-1, spacedim>(e_omega, domain_of_integration, quadrature, global_data, alpha, name),
	A(A),
	b(b)
	{
	}

	bool
	get_values_and_derivatives( const dealii::Vector<double>& 		values,
								const dealii::Point<spacedim>& 		/*x*/,
								const dealii::Tensor<1,spacedim>& 	/*n*/,
								double&								omega,
								dealii::Vector<double>&				d_omega,
								dealii::FullMatrix<double>&			d2_omega,
								const std::tuple<bool, bool, bool>	requested_quantities)
	const
	{
		const unsigned int N = A.m();
		Vector<double> A_q(N);
		A.vmult(A_q, values);

		if(get<0>(requested_quantities))
		{
			omega = 0.5 * (values * A_q ) + b * values;
		}

		if(get<1>(requested_quantities))
		{
			for(unsigned int m = 0; m < N; ++m)
				d_omega[m] = A_q[m] + b[m];
		}

		if(get<2>(requested_quantities))
		{
			d2_omega = A;
		}

		return false;
	}

};

void test_domain()
{
	const unsigned int spacedim = 3;
	const unsigned int N = 10;
	const double alpha = 0.5;

	vector<DependentField<spacedim, spacedim>> dependent_fields(N, DependentField<spacedim, spacedim>("q"));

	GlobalDataIncrementalFE<spacedim> global_data;

	FullMatrix<double> A(N,N);
	Vector<double> b(N);
	for(unsigned int m = 0; m < N; ++m)
	{
		b(m) = cos((double)m);
		for(unsigned int n = m; n < N; ++n)
			A(m, n) = A(n, m) = cos((double)m)*sin((double)n);
	}

	PsiDomain<spacedim> psi_domain(	dependent_fields,
									{},
									QGauss<spacedim>(1),
									global_data,
									A,
									b,
									alpha);

	Vector<double> e_omega(N);
	vector<Vector<double>> e_omega_ref_sets(1);
	e_omega_ref_sets[0].reinit(N);
	for(unsigned int m = 0; m < N; ++m)
	{
		e_omega[m] = cos((double)m);
		e_omega_ref_sets[0][m] = sin((double)m);
	}

	Point<spacedim> x;
	Vector<double> hidden_vars;

	double d;
	Vector<double> d1(N);
	FullMatrix<double> d2(N,N);
	psi_domain.get_h_omega(e_omega, e_omega_ref_sets, hidden_vars, x, d, d1, d2, make_tuple(true, true, true));
	psi_domain.compare_derivatives_with_numerical_derivatives(e_omega, e_omega_ref_sets, hidden_vars, x);

	Vector<double> A_q_n(N), A_q_n_p_1(N), d1_(N);
	A.vmult(A_q_n, e_omega_ref_sets[0]);
	A.vmult(A_q_n_p_1, e_omega);
	for(unsigned int m = 0; m < N; ++m)
		d1_[m] = (1.0 - alpha) * A_q_n[m] + alpha * A_q_n_p_1[m] + b[m];

	double e = 0.0;
	for(unsigned int m = 0; m < N; ++m)
		e += fabs(d1[m] - d1_[m]);
	cout << e << endl;

}

void test_interface()
{
	const unsigned int spacedim = 3;
	const unsigned int N = 10;
	const double alpha = 0.5;

	vector<DependentField<spacedim-1, spacedim>> dependent_fields(N, DependentField<spacedim-1, spacedim>("q"));

	GlobalDataIncrementalFE<spacedim> global_data;

	FullMatrix<double> A(N,N);
	Vector<double> b(N);
	for(unsigned int m = 0; m < N; ++m)
	{
		b(m) = cos((double)m);
		for(unsigned int n = m; n < N; ++n)
			A(m, n) = A(n, m) = cos((double)m)*sin((double)n);
	}

	PsiInterface<spacedim> psi_interface(	dependent_fields,
											{},
											QGauss<spacedim-1>(1),
											global_data,
											A,
											b,
											alpha);

	Vector<double> e_sigma(N);
	vector<Vector<double>> e_sigma_ref_sets(1);
	e_sigma_ref_sets[0].reinit(N);
	for(unsigned int m = 0; m < N; ++m)
	{
		e_sigma[m] = cos((double)m);
		e_sigma_ref_sets[0][m] = sin((double)m);
	}

	Point<spacedim> x;
	Vector<double> hidden_vars;
	Tensor<1, spacedim> n;

	double d;
	Vector<double> d1(N);
	FullMatrix<double> d2(N,N);
	psi_interface.get_h_sigma(e_sigma, e_sigma_ref_sets, hidden_vars, x, n, d, d1, d2, make_tuple(true, true, true));
	psi_interface.compare_derivatives_with_numerical_derivatives(e_sigma, e_sigma_ref_sets, hidden_vars, x, n);

	Vector<double> A_q_n(N), A_q_n_p_1(N), d1_(N);
	A.vmult(A_q_n, e_sigma_ref_sets[0]);
	A.vmult(A_q_n_p_1, e_sigma);
	for(unsigned int m = 0; m < N; ++m)
		d1_[m] = (1.0 - alpha) * A_q_n[m] + alpha * A_q_n_p_1[m] + b[m];

	double e = 0.0;
	for(unsigned int m = 0; m < N; ++m)
		e += fabs(d1[m] - d1_[m]);
	cout << e << endl;

}

int main()
{
	test_domain();
	test_interface();
	return 0;
}
