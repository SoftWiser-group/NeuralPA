[BugLab_Variable_Misuse]^this.matchers = this;^20^^^^^19^21^this.matchers = matchers;^[CLASS] And  [METHOD] <init> [RETURN_TYPE] List)   Matcher> matchers [VARIABLES] List  matchers  boolean  
[BugLab_Wrong_Literal]^return true;^26^^^^^23^30^return false;^[CLASS] And  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] List  matchers  Object  actual  boolean  Matcher  matcher  
[BugLab_Wrong_Literal]^return false;^29^^^^^23^30^return true;^[CLASS] And  [METHOD] matches [RETURN_TYPE] boolean   Object actual [VARIABLES] List  matchers  Object  actual  boolean  Matcher  matcher  
[BugLab_Argument_Swapping]^for  ( Iterator<Matcher> matchers = it.iterator (  ) ; it.hasNext (  ) ; )  {^34^^^^^32^41^for  ( Iterator<Matcher> it = matchers.iterator (  ) ; it.hasNext (  ) ; )  {^[CLASS] And  [METHOD] describeTo [RETURN_TYPE] void   Description description [VARIABLES] Iterator  it  List  matchers  Description  description  boolean  