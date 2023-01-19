clear;

function eq = replace_pow_2(eq)
  while(length(strfind (eq, 'pow'))>0)
    eq = regexprep (eq, 'pow\(((?:[^)(]|\((?:[^)(]|\((?:[^)(]|\([^)(]*\))*\))*\))*), 2\)', '(($1)*($1))');
  endwhile
endfunction

pkg load symbolic;

syms V_m_f D;

Idot = sym('Idot', [3 1]);
F = sym('F', [9 1]);
q = [Idot; F];
N_param = 9;
F_M = [F(1) F(2) F(3);
       F(4) F(5) F(6);
       F(7) F(8) F(9);];
C = transpose(F_M)*F_M;

Psi = simplify(V_m_f/(sym(2)*D*det(F_M)) * transpose(Idot)*C*Idot );
disp(["omega = " replace_pow_2(ccode(simplify(Psi))) ";"]);

disp(" ");

for i=1:length(q)-N_param
  dPsi_dq(i) = diff(Psi, q(i,1));
  disp(["d_omega(" num2str(i-1) ") = " replace_pow_2(ccode(simplify(dPsi_dq(i)))) ";"]);
endfor

disp(" ");

for i=1:length(q)-N_param
 for j=1:length(q)-N_param
  disp(["d2_omega(" num2str(i-1) "," num2str(j-1) ") = " replace_pow_2(ccode(simplify(diff(dPsi_dq(i), q(j,1))))) ";"]);
 endfor
endfor

for i=1:length(q)-N_param
 for j=length(q)-N_param+1:length(q)
  disp(["d2_omega(" num2str(i-1) "," num2str(j-1) ") = " replace_pow_2(ccode(simplify(diff(dPsi_dq(i), q(j,1))))) ";"]);
 endfor
endfor