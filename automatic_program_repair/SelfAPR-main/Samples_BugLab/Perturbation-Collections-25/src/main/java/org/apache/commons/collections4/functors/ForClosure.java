[BugLab_Variable_Misuse]^iCount = iCount;^70^^^^^68^72^iCount = count;^[CLASS] ForClosure  [METHOD] <init> [RETURN_TYPE] Closure)   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Variable_Misuse]^iClosure = null;^71^^^^^68^72^iClosure = closure;^[CLASS] ForClosure  [METHOD] <init> [RETURN_TYPE] Closure)   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Variable_Misuse]^if  ( iCount <= 0 || closure == null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Variable_Misuse]^if  ( count <= 0 || 4 == null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Argument_Swapping]^if  ( closure <= 0 || count == null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Wrong_Operator]^if  ( count <= 0 && closure == null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Wrong_Operator]^if  ( count < 0 || closure == null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Wrong_Operator]^if  ( count <= 0 || closure != null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Wrong_Literal]^if  ( count <= count || closure == null )  {^52^^^^^51^59^if  ( count <= 0 || closure == null )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Variable_Misuse]^if  ( iCount == 1 )  {^55^^^^^51^59^if  ( count == 1 )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Wrong_Operator]^if  ( count != 1 )  {^55^^^^^51^59^if  ( count == 1 )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Wrong_Literal]^if  ( count == count )  {^55^^^^^51^59^if  ( count == 1 )  {^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Argument_Swapping]^return new ForClosure<E> ( closure, count ) ;^58^^^^^51^59^return new ForClosure<E> ( count, closure ) ;^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Variable_Misuse]^return new ForClosure<E> ( iCount, closure ) ;^58^^^^^51^59^return new ForClosure<E> ( count, closure ) ;^[CLASS] ForClosure  [METHOD] forClosure [RETURN_TYPE] <E>   final int count Closure<? super E> closure [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  iCount  
[BugLab_Variable_Misuse]^for  ( iCountnt i = 0; i < iCount; i++ )  {^80^^^^^79^83^for  ( int i = 0; i < iCount; i++ )  {^[CLASS] ForClosure  [METHOD] execute [RETURN_TYPE] void   final E input [VARIABLES] boolean  Closure  closure  iClosure  E  input  long  serialVersionUID  int  count  i  iCount  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < count; i++ )  {^80^^^^^79^83^for  ( int i = 0; i < iCount; i++ )  {^[CLASS] ForClosure  [METHOD] execute [RETURN_TYPE] void   final E input [VARIABLES] boolean  Closure  closure  iClosure  E  input  long  serialVersionUID  int  count  i  iCount  
[BugLab_Argument_Swapping]^for  ( iCountnt i = 0; i < i; i++ )  {^80^^^^^79^83^for  ( int i = 0; i < iCount; i++ )  {^[CLASS] ForClosure  [METHOD] execute [RETURN_TYPE] void   final E input [VARIABLES] boolean  Closure  closure  iClosure  E  input  long  serialVersionUID  int  count  i  iCount  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= iCount; i++ )  {^80^^^^^79^83^for  ( int i = 0; i < iCount; i++ )  {^[CLASS] ForClosure  [METHOD] execute [RETURN_TYPE] void   final E input [VARIABLES] boolean  Closure  closure  iClosure  E  input  long  serialVersionUID  int  count  i  iCount  
[BugLab_Wrong_Literal]^for  ( int i = count; i < iCount; i++ )  {^80^^^^^79^83^for  ( int i = 0; i < iCount; i++ )  {^[CLASS] ForClosure  [METHOD] execute [RETURN_TYPE] void   final E input [VARIABLES] boolean  Closure  closure  iClosure  E  input  long  serialVersionUID  int  count  i  iCount  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < iCount; i++ )  {^80^^^^^79^83^for  ( int i = 0; i < iCount; i++ )  {^[CLASS] ForClosure  [METHOD] execute [RETURN_TYPE] void   final E input [VARIABLES] boolean  Closure  closure  iClosure  E  input  long  serialVersionUID  int  count  i  iCount  
[BugLab_Variable_Misuse]^return i;^102^^^^^101^103^return iCount;^[CLASS] ForClosure  [METHOD] getCount [RETURN_TYPE] int   [VARIABLES] boolean  Closure  closure  iClosure  long  serialVersionUID  int  count  i  iCount  
