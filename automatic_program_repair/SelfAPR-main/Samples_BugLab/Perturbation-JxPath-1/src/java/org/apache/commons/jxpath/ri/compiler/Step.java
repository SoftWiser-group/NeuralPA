[BugLab_Wrong_Operator]^if  ( predicates == null )  {^49^^^^^48^57^if  ( predicates != null )  {^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^return false;^52^^^^^48^57^return true;^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= predicates.length; i++ )  {^50^^^^^48^57^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^for  ( int i = axis; i < predicates.length; i++ )  {^50^^^^^48^57^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == predicates.length; i++ )  {^50^^^^^48^57^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < predicates.length; i++ )  {^50^^^^^48^57^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < predicates.length; i++ )  {^50^^^^^48^57^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^return true;^56^^^^^48^57^return false;^[CLASS] Step  [METHOD] isContextDependent [RETURN_TYPE] boolean   [VARIABLES] boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Variable_Misuse]^buffer.append ( iToString ( axis )  ) ;^89^^^^^81^92^buffer.append ( axisToString ( axis )  ) ;^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Variable_Misuse]^buffer.append ( iToString ( axis )  ) ;^89^^^^^74^104^buffer.append ( axisToString ( axis )  ) ;^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Operator]^if  ( predicates == null )  {^94^^^^^79^109^if  ( predicates != null )  {^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Operator]^for  ( int i = 0; i == predicates.length; i++ )  {^95^^^^^80^110^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^for  ( int i = -1; i < predicates.length; i++ )  {^95^^^^^80^110^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^for  ( int i = axis; i < predicates.length; i++ )  {^95^^^^^80^110^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= predicates.length; i++ )  {^95^^^^^80^110^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
[BugLab_Wrong_Literal]^for  ( int i = 1; i < predicates.length; i++ )  {^95^^^^^80^110^for  ( int i = 0; i < predicates.length; i++ )  {^[CLASS] Step  [METHOD] toString [RETURN_TYPE] String   [VARIABLES] StringBuffer  buffer  boolean  Expression[]  predicates  int  axis  i  NodeTest  nodeTest  
