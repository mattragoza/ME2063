function Yout = FOM_rhs(t,Y)
global  FEM_K  FEM_F EID
Yout = -(FEM_K*Y) + FEM_F;
Yout(EID') = 0;
end