[BugLab_Argument_Swapping]^return attribute.getAttribute ( pageContext, PageContext.PAGE_SCOPE ) ;^47^^^^^46^48^return pageContext.getAttribute ( attribute, PageContext.PAGE_SCOPE ) ;^[CLASS] PageScopeContext  [METHOD] getAttribute [RETURN_TYPE] Object   String attribute [VARIABLES] PageContext  pageContext  String  attribute  boolean  
[BugLab_Argument_Swapping]^pageContext.setAttribute ( value, attribute, PageContext.PAGE_SCOPE ) ;^51^^^^^50^52^pageContext.setAttribute ( attribute, value, PageContext.PAGE_SCOPE ) ;^[CLASS] PageScopeContext  [METHOD] setAttribute [RETURN_TYPE] void   String attribute Object value [VARIABLES] PageContext  pageContext  Object  value  String  attribute  boolean  