[BugLab_Variable_Misuse]^super ( name, quadLen ) ;^15^^^^^13^24^super ( name, hash ) ;^[CLASS] NameN  [METHOD] <init> [RETURN_TYPE] String,int,int[],int)   String name int hash int[] quads int quadLen [VARIABLES] int[]  mQuads  quads  String  name  boolean  int  hash  mQuadLen  quadLen  
[BugLab_Argument_Swapping]^super ( hash, name ) ;^15^^^^^13^24^super ( name, hash ) ;^[CLASS] NameN  [METHOD] <init> [RETURN_TYPE] String,int,int[],int)   String name int hash int[] quads int quadLen [VARIABLES] int[]  mQuads  quads  String  name  boolean  int  hash  mQuadLen  quadLen  
[BugLab_Variable_Misuse]^if  ( mQuadLen < 3 )  {^19^^^^^13^24^if  ( quadLen < 3 )  {^[CLASS] NameN  [METHOD] <init> [RETURN_TYPE] String,int,int[],int)   String name int hash int[] quads int quadLen [VARIABLES] int[]  mQuads  quads  String  name  boolean  int  hash  mQuadLen  quadLen  
[BugLab_Wrong_Operator]^if  ( quadLen <= 3 )  {^19^^^^^13^24^if  ( quadLen < 3 )  {^[CLASS] NameN  [METHOD] <init> [RETURN_TYPE] String,int,int[],int)   String name int hash int[] quads int quadLen [VARIABLES] int[]  mQuads  quads  String  name  boolean  int  hash  mQuadLen  quadLen  
[BugLab_Variable_Misuse]^mQuads = mQuads;^22^^^^^13^24^mQuads = quads;^[CLASS] NameN  [METHOD] <init> [RETURN_TYPE] String,int,int[],int)   String name int hash int[] quads int quadLen [VARIABLES] int[]  mQuads  quads  String  name  boolean  int  hash  mQuadLen  quadLen  
[BugLab_Variable_Misuse]^mQuadLen = mQuadLen;^23^^^^^13^24^mQuadLen = quadLen;^[CLASS] NameN  [METHOD] <init> [RETURN_TYPE] String,int,int[],int)   String name int hash int[] quads int quadLen [VARIABLES] int[]  mQuads  quads  String  name  boolean  int  hash  mQuadLen  quadLen  
[BugLab_Wrong_Literal]^public boolean equals ( int quad )  { return true; }^28^^^^^23^33^public boolean equals ( int quad )  { return false; }^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int quad [VARIABLES] int[]  mQuads  quads  int  hash  mQuadLen  quad  quadLen  boolean  
[BugLab_Wrong_Literal]^public boolean equals ( int quad1, int quad2 )  { return true; }^32^^^^^27^37^public boolean equals ( int quad1, int quad2 )  { return false; }^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int quad1 int quad2 [VARIABLES] int[]  mQuads  quads  int  hash  mQuadLen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Variable_Misuse]^if  ( quadLen != mQuadLen )  {^37^^^^^22^52^if  ( qlen != mQuadLen )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Variable_Misuse]^if  ( qlen != quad2 )  {^37^^^^^22^52^if  ( qlen != mQuadLen )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Argument_Swapping]^if  ( mQuadLen != qlen )  {^37^^^^^22^52^if  ( qlen != mQuadLen )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Operator]^if  ( qlen >= mQuadLen )  {^37^^^^^22^52^if  ( qlen != mQuadLen )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Literal]^return true;^38^^^^^23^53^return false;^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Operator]^if  ( quads[i] == mQuads[i] )  {^62^^^^^47^77^if  ( quads[i] != mQuads[i] )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Literal]^return true;^63^^^^^48^78^return false;^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Variable_Misuse]^for  ( quadLennt i = 0; i < qlen; ++i )  {^61^^^^^46^76^for  ( int i = 0; i < qlen; ++i )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Variable_Misuse]^for  ( int i = 0; i < quad2; ++i )  {^61^^^^^46^76^for  ( int i = 0; i < qlen; ++i )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Operator]^for  ( int i = 0; i <= qlen; ++i )  {^61^^^^^46^76^for  ( int i = 0; i < qlen; ++i )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Variable_Misuse]^if  ( mQuads[i] != mQuads[i] )  {^62^^^^^47^77^if  ( quads[i] != mQuads[i] )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Operator]^if  ( quads[i] >= mQuads[i] )  {^62^^^^^47^77^if  ( quads[i] != mQuads[i] )  {^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
[BugLab_Wrong_Literal]^return false;^66^^^^^51^81^return true;^[CLASS] NameN  [METHOD] equals [RETURN_TYPE] boolean   int[] quads int qlen [VARIABLES] int[]  mQuads  quads  int  hash  i  mQuadLen  qlen  quad  quad1  quad2  quadLen  boolean  
